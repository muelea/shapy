from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import time

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..structures import (Keypoints2D, Keypoints3D,
                          GlobalRot, BodyPose, HandPose, JawPose,
                          Betas, Joints)
from ..utils import (
    get_part_idxs,
    create_flip_indices,
    scale_to_bbox_size,
    keyps_to_bbox, bbox_to_center_scale,
    KEYPOINT_NAMES_DICT, KEYPOINT_PARTS,
)

from human_shape.utils import read_img, binarize

FOLDER_MAP_FNAME = 'folder_map.pkl'


class Human36M(dutils.Dataset):
    def __init__(self,
                 data_folder='data/h36m',
                 img_folder='images',
                 keyp_folder='keypoints',
                 split='train',
                 use_face=True, use_hands=True, use_face_contour=False,
                 model_type='smplx',
                 dtype=torch.float32,
                 use_joint_conf=True,
                 metrics=None,
                 transforms=None,
                 conf_thresh=0.3,
                 binarization=True,
                 return_shape=False,
                 return_full_pose=False,
                 **kwargs):
        super(Human36M, self).__init__()

        if metrics is None:
            metrics = []
        self.metrics = metrics

        self.split = split
        self.is_train = 'train' in split
        self.return_shape = return_shape
        self.return_full_pose = return_full_pose

        self.data_folder = osp.expandvars(osp.expanduser(data_folder))
        self.binarization = binarization

        folder_map_fname = osp.join(self.data_folder, split, FOLDER_MAP_FNAME)
        self.use_folder_split = osp.exists(folder_map_fname)
        if self.use_folder_split:
            self.data_folder = osp.join(self.data_folder, split)
            with open(folder_map_fname, 'rb') as f:
                data_dict = pickle.load(f)

            self.items_per_folder = max(data_dict.values())

        self.img_folder = osp.join(self.data_folder, img_folder)
        self.keyp_folder = osp.join(self.data_folder, keyp_folder)

        self.transforms = transforms
        self.dtype = dtype

        self.use_face = use_face
        self.use_hands = use_hands
        self.use_face_contour = use_face_contour
        self.model_type = model_type
        self.use_joint_conf = use_joint_conf
        self.conf_thresh = conf_thresh

        fname = ('h36m_single_train.npz' if self.is_train else
                 'h36m_single_val.npz')
        annotations_fn = osp.join(self.data_folder, fname)

        data = np.load(annotations_fn)
        data_dict = {key: data[key] for key in data}

        self.img_names = np.asarray(data_dict['imgname'])
        self.bbox = np.asarray(data_dict['bbox'])
        self.center = np.asarray(data_dict['center'])
        self.scale = np.asarray(data_dict['scale'])
        self.pose = np.asarray(data_dict['pose'])
        if self.return_shape:
            self.betas = np.asarray(data_dict['shape']).astype(np.float32)
        self.joints14 = np.asarray(data_dict['S'])
        self.keypoints2D = np.asarray(data_dict['part'])

        self.num_items = self.joints14.shape[0]

        self.source = 'spin'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source]
        self.flip_indices = create_flip_indices(self.keypoint_names)
        self.h36m_to_j14 = np.asarray(list(range(13)) + [18])

    def __repr__(self):
        return 'Human3.6m( \n\t Split: {}\n)'.format(self.split)

    def name(self):
        return 'Human3.6m/{}'.format(self.split)

    def get_num_joints(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_items

    def get_elements_per_index(self):
        return 1

    def only_2d(self):
        return False

    def __getitem__(self, index):
        if self.use_folder_split:
            folder_idx = index // self.items_per_folder
            img_path = osp.join(self.img_folder,
                                'folder_{:010d}'.format(folder_idx),
                                '{:010d}.jpg'.format(index))
            keyp_path = osp.join(self.keyp_folder,
                                 'folder_{:010d}'.format(folder_idx),
                                 '{:010d}_keypoints.json'.format(index))
        else:
            img_fn = self.img_names[index].decode('utf-8')
            fname, _ = osp.splitext(img_fn)
            tokens = fname.split('_')
            subject = tokens[0]
            action = '_'.join(tokens[1:3])

            img_path = osp.join(self.img_folder, subject, action, img_fn)

            fname, _ = osp.splitext(img_fn)

        img = read_img(img_path)

        pose = self.pose[index].reshape(-1, 3).astype(np.float32)
        global_rot = pose[0].reshape(-1)
        if self.return_full_pose:
            body_pose = pose[1:].reshape(-1)
        else:
            body_pose = pose[1:22, :].reshape(-1)

        keypoints2d = self.keypoints2D[index].copy()
        keypoints2d[:, -1] = np.clip(keypoints2d[:, -1], 0, 1)

        # Only keep the points with confidence above a threshold
        if self.conf_thresh > 0:
            keypoints2d[keypoints2d[:, -1] < self.conf_thresh, -1] = 0

        # If we don't want to use the confidence scores as weights for the loss
        # then set those above the conf thresh to 1
        dtype = keypoints2d.dtype
        if self.binarization:
            keypoints2d[:, -1] = binarize(
                keypoints2d[:, -1], self.conf_thresh, dtype=dtype)

        target = Keypoints2D(
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            source=self.source,
            flip_axis=0,
            dtype=self.dtype)
        bbox = self.bbox[index]
        bbox_size = max(bbox[3] - bbox[1], bbox[2] - bbox[0])
        target.add_field('center', self.center[index].astype(np.float32))
        target.add_field('scale', self.scale[index])
        target.add_field('bbox_size', bbox_size)
        target.add_field('dset_scale_factor', 1.0)

        target.add_field('orig_center', self.center[index].astype(np.float32))
        target.add_field('orig_bbox_size', bbox_size)

        global_rot_field = GlobalRot(global_rot=global_rot)
        target.add_field('global_rot', global_rot_field)
        body_pose_field = BodyPose(body_pose=body_pose)
        target.add_field('body_pose', body_pose_field)
        if self.return_shape:
            betas = self.betas[index]
            target.add_field('betas', Betas(betas))

        target.add_field('name', self.name())
        if not self.is_train:
            joints = self.joints14[index, self.h36m_to_j14, :3]
            joints = Joints(joints)
            target.add_field('joints', joints)

        if self.transforms is not None:
            full_img, cropped_image, cropped_target = self.transforms(
                img, target, force_flip=False)

        return full_img, cropped_image, cropped_target, index


class Human36MX(dutils.Dataset):
    def __init__(self, data_folder='data/h36mx',
                 img_folder='images',
                 keyp_folder='keypoints',
                 annotations_fn='human36m_smplx_train_rate_4.npz',
                 num_betas=10,
                 split='train',
                 use_face=True, use_hands=True, use_face_contour=False,
                 model_type='smplx',
                 dtype=torch.float32,
                 metrics=None,
                 transforms=None,
                 binarization=True,
                 return_shape=False,
                 return_full_pose=False,
                 return_gender=True,
                 **kwargs):
        super(Human36MX, self).__init__()

        if metrics is None:
            metrics = []
        self.metrics = metrics

        self.split = split
        self.is_train = 'train' in split
        self.return_shape = return_shape
        self.return_full_pose = return_full_pose
        self.return_gender = return_gender

        self.data_folder = osp.expandvars(osp.expanduser(data_folder))
        self.binarization = binarization

        folder_map_fname = osp.join(self.data_folder, split, FOLDER_MAP_FNAME)
        self.use_folder_split = osp.exists(folder_map_fname)
        if self.use_folder_split:
            self.data_folder = osp.join(self.data_folder, split)
            with open(folder_map_fname, 'rb') as f:
                data_dict = pickle.load(f)

            self.items_per_folder = max(data_dict.values())

        self.img_folder = osp.join(self.data_folder, img_folder)
        self.keyp_folder = osp.join(self.data_folder, keyp_folder)

        self.transforms = transforms
        self.dtype = dtype

        self.use_face = use_face
        self.use_hands = use_hands
        self.use_face_contour = use_face_contour
        self.model_type = model_type

        annotations_fn = osp.expandvars(annotations_fn)
        if not self.is_train:
            annot_folder, fname = osp.split(annotations_fn)
            annotations_fn = osp.join(
                annot_folder, fname.replace('train', split))
        logger.info(f'Loading H3.6M data from: {annotations_fn}')
        data = np.load(annotations_fn)
        data_dict = {key: data[key] for key in data}

        self.img_names = np.asarray(data_dict['imgname'])
        self.center = np.asarray(data_dict['center']).astype(np.float32)
        self.scale = np.asarray(data_dict['scale']).astype(np.float32)
        self.global_rot = np.asarray(
            data_dict['global_pose']).astype(np.float32)
        self.body_pose = np.asarray(data_dict['body_pose']).astype(np.float32)
        self.left_hand_pose = np.asarray(
            data_dict['left_hand_pose']).astype(np.float32)
        self.right_hand_pose = np.asarray(
            data_dict['right_hand_pose']).astype(np.float32)
        self.jaw_pose = np.asarray(data_dict['jaw_pose']).astype(np.float32)
        self.translation = data_dict['translation'].copy().astype(np.float32)

        self.betas = np.asarray(data_dict['betas']).astype(np.float32)
        self.betas = self.betas[:, :num_betas].copy()
        if return_gender and 'gender' in data_dict:
            self.gender = np.asarray(data_dict['gender'])

        self.keypoints3D = np.asarray(data_dict['S'])
        self.keypoints2D = np.asarray(data_dict['part'])

        self.num_items = self.keypoints3D.shape[0]

        self.source = 'raw_h36m'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source]
        self.flip_indices = create_flip_indices(self.keypoint_names)

        self.h36m_to_j14 = np.asarray(list(range(13)) + [18])

    def __repr__(self):
        return f'Human3.6mX( \n\t Split: {self.split}\n)'

    def name(self):
        return f'Human3.6mX/{self.split}'

    def __len__(self):
        return self.num_items

    def get_elements_per_index(self):
        return 1

    def only_2d(self):
        return False

    def fname_to_path(self, img_fn):
        if not isinstance(img_fn, str):
            img_fn = img_fn.decode('utf-8')
        _, img_fn = osp.split(img_fn)
        tokens = osp.split(img_fn)[1].split('_')
        fname = img_fn.replace('.jpg', '')

        tokens = fname.split('_')
        subject = tokens[0]
        if len(tokens) == 3:
            action = tokens[1]
        elif len(tokens) == 4:
            action = '_'.join(tokens[1:3])
        if 'WalkingDog' in action:
            action = action.replace('WalkingDog', 'WalkDog')
        if 'TakingPhoto' in action:
            action = action.replace('TakingPhoto', 'Photo')
        img_path = osp.join(self.img_folder, subject,
                            action, f'{tokens[-1]}.jpg')
        return img_path

    def __getitem__(self, index):
        if self.use_folder_split:
            folder_idx = index // self.items_per_folder
            img_path = osp.join(self.img_folder,
                                'folder_{:010d}'.format(folder_idx),
                                '{:010d}.jpg'.format(index))
        else:
            img_fn = self.img_names[index]
            img_path = self.fname_to_path(img_fn)
            fname, _ = osp.splitext(img_fn)

        img = read_img(img_path)

        global_rot = self.global_rot[index].reshape(-1)
        body_pose = self.body_pose[index].reshape(-1)

        keypoints2d = self.keypoints2D[index].copy()
        keypoints3d = self.keypoints3D[index].copy()
        keypoints2d = np.concatenate(
            [keypoints2d, np.ones_like(keypoints2d[:, [-1]])], axis=-1
        )
        keypoints3d = np.concatenate(
            [keypoints3d, np.ones_like(keypoints2d[:, [-1]])], axis=-1
        )

        target = Keypoints2D(
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source,
            dtype=self.dtype)
        bbox_size = scale_to_bbox_size(self.scale[index])

        keypoints_hd = Keypoints2D(
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source,
            apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        target.add_field('center', self.center[index])
        target.add_field('scale', self.scale[index])
        target.add_field('bbox_size', bbox_size)
        target.add_field('dset_scale_factor', 1.0)
        target.add_field('orig_bbox_size', bbox_size)
        target.add_field('orig_center', self.center[index])

        keyps3d_target = Keypoints3D(
            keypoints3d, img.shape, flip_indices=self.flip_indices,
            source=self.source,
            flip_axis=0, dtype=self.dtype)
        target.add_field('keypoints3d', keyps3d_target)

        global_rot_field = GlobalRot(global_rot=global_rot)
        target.add_field('global_rot', global_rot_field)
        body_pose_field = BodyPose(body_pose=body_pose)
        target.add_field('body_pose', body_pose_field)
        if self.return_shape:
            betas = self.betas[index]
            target.add_field('betas', Betas(betas))
        if self.return_full_pose:
            hand_pose_field = HandPose(
                left_hand_pose=self.left_hand_pose[index].reshape(-1),
                right_hand_pose=self.right_hand_pose[index].reshape(-1),
            )
            target.add_field('hand_pose', hand_pose_field)
            jaw_pose_field = JawPose(
                jaw_pose=self.jaw_pose[index].reshape(-1),
            )
            target.add_field('jaw_pose', jaw_pose_field)

        if not self.is_train:
            joints = self.joints14[index, self.h36m_to_j14, :3]
            joints = Joints(joints)
            target.add_field('joints', joints)
        if self.return_gender:
            gender = str(self.gender[index])
            if gender == 'male' or gender == 'female':
                gender = 'M' if gender == 'male' else 'F'
                target.add_field('gender', gender)
            else:
                target.add_field('gender', '')

        if self.transforms is not None:
            full_img, cropped_image, cropped_target = self.transforms(
                img, target, force_flip=False)
        target.add_field('name', self.name())

        return full_img, cropped_image, cropped_target, index
