import sys
import os
import os.path as osp

import pickle

import tqdm
import time

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..structures import Keypoints2D, Joints, Vertices

from ..utils import (
    get_part_idxs,
    create_flip_indices,
    keyps_to_bbox, bbox_to_center_scale,
    KEYPOINT_NAMES_DICT, KEYPOINT_PARTS,
)
from human_shape.utils import read_img, binarize

FOLDER_MAP_FNAME = 'folder_map.pkl'


class ThreeDPW(dutils.Dataset):
    def __init__(self, data_folder='data/3dpw',
                 img_folder='',
                 seq_folder='sequenceFiles',
                 param_folder='smplx_npz_data',
                 split='val',
                 use_face=True, use_hands=True, use_face_contour=False,
                 model_type='smplx',
                 dtype=torch.float32,
                 vertex_folder='smplx_vertices',
                 return_vertices=True,
                 metrics=None,
                 transforms=None,
                 body_thresh=0.3,
                 binarization=True,
                 min_visible=6,
                 **kwargs):
        super(ThreeDPW, self).__init__()

        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.binarization = binarization
        self.return_vertices = return_vertices

        self.split = split
        self.is_train = 'train' in split

        self.data_folder = osp.expandvars(osp.expanduser(data_folder))
        seq_path = osp.join(self.data_folder, seq_folder)
        if self.split == 'train':
            seq_split_path = osp.join(seq_path, 'train')
            npz_fn = osp.join(self.data_folder, param_folder, '3dpw_train.npz')
        elif self.split == 'val':
            seq_split_path = osp.join(seq_path, 'validation')
            npz_fn = osp.join(
                self.data_folder, param_folder, '3dpw_validation.npz')
        elif self.split == 'test':
            seq_split_path = osp.join(seq_path, 'test')
            npz_fn = osp.join(self.data_folder, param_folder, '3dpw_test.npz')

        self.vertex_folder = osp.join(
            self.data_folder, vertex_folder, self.split)

        self.img_folder = osp.join(self.data_folder, img_folder)
        folder_map_fname = osp.expandvars(
            osp.join(self.img_folder, split, FOLDER_MAP_FNAME))
        self.use_folder_split = osp.exists(folder_map_fname)
        if self.use_folder_split:
            with open(folder_map_fname, 'rb') as f:
                data_dict = pickle.load(f)
            self.items_per_folder = max(data_dict.values())
            self.img_folder = osp.join(self.img_folder, split)

        data_dict = np.load(npz_fn)

        if 'cam_intrinsics' in data_dict:
            self.cam_intrinsics = data_dict['cam_intrinsics']

        self.img_paths = np.asarray(data_dict['img_paths'])

        idxs = np.arange(len(self.img_paths))
        self.idxs = idxs
        self.img_paths = self.img_paths[idxs]

        if 'keypoints2d' in data_dict:
            self.keypoints2d = np.asarray(
                data_dict['keypoints2d']).astype(np.float32)[idxs]
        elif 'keypoints2D' in data_dict:
            self.keypoints2d = np.asarray(
                data_dict['keypoints2D']).astype(np.float32)[idxs]
        else:
            raise KeyError(f'Keypoints2D not in 3DPW {split} dictionary')
        self.joints3d = np.asarray(
            data_dict['joints3d']).astype(np.float32)[idxs]
        self.num_items = len(self.img_paths)
        self.pids = np.asarray(data_dict['pid'], dtype=np.int32)
        self.center = np.asarray(
            data_dict['center'], dtype=np.float32)[idxs]
        self.scale = np.asarray(
            data_dict['scale'], dtype=np.float32)[idxs]
        self.bbox_size = np.asarray(
            data_dict['bbox_size'], dtype=np.float32)[idxs]

        self.transforms = transforms
        self.dtype = dtype

        self.use_face = use_face
        self.use_hands = use_hands
        self.use_face_contour = use_face_contour
        self.model_type = model_type
        self.body_thresh = body_thresh

        self.source = '3dpw'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source]
        self.flip_indices = create_flip_indices(self.keypoint_names)
        idxs_dict = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)

        body_idxs = idxs_dict['body']
        self.body_idxs = np.asarray(body_idxs)

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return f'3DPW( \n\t Split: {self.split}\n)'.format(self.split)

    def name(self):
        return f'3DPW/{self.split}'

    def __len__(self):
        return self.num_items

    def only_2d(self):
        return False

    def __getitem__(self, index):
        img_fn = self.img_paths[index]

        if self.use_folder_split:
            folder_idx = (index + self.idxs[0]) // self.items_per_folder
            img_fn = osp.join(self.img_folder,
                              'folder_{:010d}'.format(folder_idx),
                              f'{index + self.idxs[0]:010d}.jpg')
        img = read_img(img_fn)

        keypoints2d = self.keypoints2d[index, :].copy()
        keypoints2d[:, -1] = np.clip(keypoints2d[:, -1], 0, 1)

        body_conf = keypoints2d[self.body_idxs, -1]
        if self.body_thresh > 0:
            body_conf[body_conf < self.body_thresh] = 0.0

        if self.binarization:
            body_conf = binarize(
                body_conf, self.body_thresh, keypoints2d.dtype)

        center = self.center[index]
        scale = self.scale[index]
        bbox_size = self.bbox_size[index]
        #  keypoints = output_keypoints2d[:, :-1]
        #  conf = output_keypoints2d[:, -1]
        target = Keypoints2D(
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            source=self.source,
            flip_axis=0,
            dtype=self.dtype)
        target.add_field('center', center)
        target.add_field('orig_center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('orig_bbox_size', bbox_size)

        keypoints_hd = Keypoints2D(
            keypoints2d, img.shape, flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source,
            apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        target.add_field('filename', self.img_paths[index])

        head, fname = osp.split(self.img_paths[index])
        _, seq_name = osp.split(head)
        target.add_field('fname', f'{seq_name}/{fname}_{self.pids[index]}')

        if self.return_vertices:
            vertex_fname = osp.join(
                self.vertex_folder,
                f'{index + self.idxs[0]:06d}.npy')
            vertices = np.load(vertex_fname)

            vertex_field = Vertices(vertices.reshape(-1, 3))
            target.add_field('vertices', vertex_field)

            intrinsics = self.cam_intrinsics[index]
            target.add_field('intrinsics', intrinsics)

        if not self.is_train:
            joints3d = self.joints3d[index]
            joints = Joints(joints3d[:14])
            target.add_field('joints14', joints)

            if hasattr(self, 'v_shaped'):
                v_shaped = self.v_shaped[index]
                target.add_field('v_shaped', Vertices(v_shaped))

        if self.transforms is not None:
            img, cropped_image, target = self.transforms(
                img, target, force_flip=False)

        return img, cropped_image, target, index
