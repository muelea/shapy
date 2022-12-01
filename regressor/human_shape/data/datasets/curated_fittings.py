import sys
import os
import os.path as osp

import pickle
import time

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..structures import (Keypoints2D,
                          Betas, Expression, GlobalRot, BodyPose,
                          HandPose, JawPose, Vertices,
                          BoundingBox)
from ..utils import (
    get_part_idxs,
    create_flip_indices,
    keyps_to_bbox, bbox_to_center_scale,
    KEYPOINT_NAMES_DICT, KEYPOINT_PARTS,
)

from human_shape.utils import nand, read_img, binarize

FOLDER_MAP_FNAME = 'folder_map.pkl'


class CuratedFittings(dutils.Dataset):
    def __init__(self, data_folder='data/curated_fits',
                 split='train',
                 img_folder='',
                 use_face=True,
                 use_hands=True,
                 head_only=False,
                 hand_only=False,
                 return_gender=True,
                 model_type='smplx',
                 keyp_format='coco25',
                 dtype=torch.float32,
                 metrics=None,
                 transforms=None,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 min_hand_keypoints=8,
                 min_head_keypoints=8,
                 binarization=True,
                 return_params=True,
                 vertex_folder='vertices',
                 vertex_flip_correspondences='',
                 body_dset_factor=1.2,
                 hand_dset_factor=2.0,
                 head_dset_factor=2.0,
                 **kwargs):
        super(CuratedFittings, self).__init__()

        assert nand(head_only, hand_only), (
            'Hand only and head only can\'t be True at the same time')
        self.split = split

        self.binarization = binarization
        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.min_hand_keypoints = min_hand_keypoints
        self.min_head_keypoints = min_head_keypoints

        if 'test' in split:
            split = 'val'
        self.is_train = 'train' in split
        self.return_params = return_params
        self.return_gender = return_gender

        self.head_only = head_only
        self.hand_only = hand_only

        self.body_dset_factor = body_dset_factor
        self.head_dset_factor = head_dset_factor
        self.hand_dset_factor = hand_dset_factor

        data_folder = osp.expandvars(osp.expanduser(data_folder))
        self.data_folder = osp.join(data_folder, f'{split}.npz')
        self.transforms = transforms
        self.dtype = dtype

        vertex_flip_correspondences = osp.expandvars(
            vertex_flip_correspondences)
        err_msg = (
            'Vertex flip correspondences path does not exist:' +
            f' {vertex_flip_correspondences}'
        )
        assert osp.exists(vertex_flip_correspondences), err_msg
        flip_data = np.load(vertex_flip_correspondences)
        self.bc = flip_data['bc']
        self.closest_faces = flip_data['closest_faces']

        self.img_folder = osp.expandvars(osp.join(img_folder, split))
        folder_map_fname = osp.expandvars(
            osp.join(self.img_folder, FOLDER_MAP_FNAME))
        self.use_folder_split = osp.exists(folder_map_fname)
        if self.use_folder_split:
            with open(folder_map_fname, 'rb') as f:
                data_dict = pickle.load(f)
            self.items_per_folder = max(data_dict.values())

        self.use_face = use_face
        self.use_hands = use_hands
        self.model_type = model_type
        self.keyp_format = keyp_format
        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh

        data = np.load(self.data_folder, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}

        self.betas = data['betas'].astype(np.float32)
        self.expression = data['expression'].astype(np.float32)
        self.keypoints2D = data['keypoints2D'].astype(np.float32)[:, :-2]
        self.pose = data['pose'].astype(np.float32)
        self.translation = data['translation'].astype(np.float32)
        self.pelvis = data['pelvis'].astype(np.float32)
        self.img_fns = np.asarray(data['img_fns'], dtype=np.string_)
        self.indices = None
        if 'indices' in data:
            self.indices = np.asarray(data['indices'], dtype=np.int64)
        self.is_right = None
        if 'is_right' in data:
            self.is_right = np.asarray(data['is_right'], dtype=np.bool_)
        if 'dset_name' in data:
            self.dset_name = np.asarray(data['dset_name'], dtype=np.string_)
        self.vertex_folder = osp.join(data_folder, vertex_folder, split)

        if self.return_gender:
            assert 'gender' in data, (
                f'Requested gender labels, but they are not there'
                f' The labels found are: {data.keys()}'
            )
            self.gender = np.asarray(data['gender'], dtype=np.string_)

        if self.use_folder_split:
            self.num_items = sum(data_dict.values())
        else:
            self.num_items = self.pose.shape[0]

        data.clear()
        del data

        self.keypoint_names = KEYPOINT_NAMES_DICT['openpose25_v1']
        self.flip_indices = create_flip_indices(self.keypoint_names)

        idxs_dict = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        head_idxs = idxs_dict['head']

        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.head_idxs = np.asarray(head_idxs)

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return self.name()

    def name(self):
        return f'Curated Fittings/{self.split}'

    def __len__(self):
        return self.num_items

    def only_2d(self):
        return False

    def __getitem__(self, index):
        img_index = index
        if self.indices is not None:
            img_index = self.indices[index]

        if self.use_folder_split:
            folder_idx = img_index // self.items_per_folder
            file_idx = img_index

        is_right = None
        if self.is_right is not None:
            is_right = self.is_right[index]

        pose = self.pose[index].copy()
        betas = self.betas[index]
        expression = self.expression[index]

        eye_offset = 0 if pose.shape[0] == 53 else 2
        global_rot = pose[0].reshape(-1)

        body_pose = pose[1:22, :].reshape(-1)
        jaw_pose = pose[22].reshape(-1)
        left_hand_pose = pose[
            23 + eye_offset:23 + eye_offset + 15].reshape(-1)
        right_hand_pose = pose[23 + 15 + eye_offset:].reshape(-1)

        keypoints2d = self.keypoints2D[index].copy()

        if self.use_folder_split:
            img_fn = osp.join(self.img_folder,
                              'folder_{:010d}'.format(folder_idx),
                              '{:010d}.jpg'.format(file_idx))
        else:
            img_fn = self.img_fns[index].decode('utf-8')

        img = read_img(img_fn)

        keypoints2d[:, -1] = np.clip(keypoints2d[:, -1], 0, 1)
        body_conf = keypoints2d[self.body_idxs, -1]
        if self.body_thresh > 0:
            body_conf[body_conf < self.body_thresh] = 0.0

        # Only keep the points with confidence above a threshold
        if self.head_only or self.hand_only:
            body_conf[:] = 0.0

        left_hand_conf = keypoints2d[self.left_hand_idxs, -1]
        right_hand_conf = keypoints2d[self.right_hand_idxs, -1]
        if self.hand_thresh > 0:
            left_hand_conf[left_hand_conf < self.hand_thresh] = 0.0
            right_hand_conf[right_hand_conf < self.hand_thresh] = 0.0

        face_conf = keypoints2d[self.face_idxs, -1]
        if self.face_thresh > 0:
            face_conf[face_conf < self.face_thresh] = 0.0

        if self.head_only:
            left_hand_conf[:] = 0.0
            right_hand_conf[:] = 0.0

        if self.hand_only:
            face_conf[:] = 0.0
            if is_right:
                left_hand_conf[:] = 0
            else:
                right_hand_conf[:] = 0

        if self.binarization:
            body_conf = binarize(
                body_conf, self.body_thresh, keypoints2d.dtype)
            left_hand_conf = binarize(
                left_hand_conf, self.hand_thresh, keypoints2d.dtype)
            right_hand_conf = binarize(
                right_hand_conf, self.hand_thresh, keypoints2d.dtype)
            face_conf = binarize(
                face_conf, self.face_thresh, keypoints2d.dtype)

        # Copy the updated confidence scores back to the keypoints
        keypoints2d[self.body_idxs, -1] = body_conf
        keypoints2d[self.left_hand_idxs, -1] = left_hand_conf
        keypoints2d[self.right_hand_idxs, -1] = right_hand_conf
        keypoints2d[self.face_idxs, -1] = face_conf

        target = Keypoints2D(
            keypoints2d, img.shape, flip_indices=self.flip_indices,
            flip_axis=0,
            source='openpose25_v1',
            dtype=self.dtype)

        if self.head_only:
            keypoints = keypoints2d[self.head_idxs, :-1]
            conf = keypoints2d[self.head_idxs, -1]
        elif self.hand_only:
            keypoints = keypoints2d[self.hand_idxs, :-1]
            conf = keypoints2d[self.hand_idxs, -1]
        else:
            keypoints = keypoints2d[:, :-1]
            conf = keypoints2d[:, -1]

        left_hand_bbox = keyps_to_bbox(
            keypoints2d[self.left_hand_idxs, :-1],
            keypoints2d[self.left_hand_idxs, -1],
            img_size=img.shape, scale=1.5)
        left_hand_bbox_target = BoundingBox(left_hand_bbox, img.shape)
        has_left_hand = (keypoints2d[self.left_hand_idxs, -1].sum() >
                         self.min_hand_keypoints)
        if has_left_hand:
            target.add_field('left_hand_bbox', left_hand_bbox_target)
            target.add_field(
                'orig_left_hand_bbox',
                BoundingBox(left_hand_bbox, img.shape, transform=False))

        right_hand_bbox = keyps_to_bbox(
            keypoints2d[self.right_hand_idxs, :-1],
            keypoints2d[self.right_hand_idxs, -1],
            img_size=img.shape, scale=1.5)
        right_hand_bbox_target = BoundingBox(right_hand_bbox, img.shape)
        has_right_hand = (keypoints2d[self.right_hand_idxs, -1].sum() >
                          self.min_hand_keypoints)
        if has_right_hand:
            target.add_field('right_hand_bbox', right_hand_bbox_target)
            target.add_field(
                'orig_right_hand_bbox',
                BoundingBox(right_hand_bbox, img.shape, transform=False))

        head_bbox = keyps_to_bbox(
            keypoints2d[self.head_idxs, :-1],
            keypoints2d[self.head_idxs, -1],
            img_size=img.shape, scale=1.2)
        head_bbox_target = BoundingBox(head_bbox, img.shape)
        has_head = (keypoints2d[self.head_idxs, -1].sum() >
                    self.min_head_keypoints)
        if has_head:
            target.add_field('head_bbox', head_bbox_target)
            target.add_field(
                'orig_head_bbox',
                BoundingBox(head_bbox, img.shape, transform=False))

        if self.head_only:
            dset_scale_factor = self.head_dset_factor
        elif self.hand_only:
            dset_scale_factor = self.hand_dset_factor
        else:
            dset_scale_factor = self.body_dset_factor
        center, scale, bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.shape),
            dset_scale_factor=dset_scale_factor,
        )

        keypoints_hd = Keypoints2D(
            keypoints2d, img.shape, flip_indices=self.flip_indices,
            flip_axis=0,
            source='openpose25_v1',
            apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('dset_scale_factor', dset_scale_factor)
        target.add_field('orig_center', center)
        target.add_field('orig_bbox_size', bbox_size)

        if self.return_params:
            betas_field = Betas(betas=betas)
            target.add_field('betas', betas_field)

            expression_field = Expression(expression=expression)
            target.add_field('expression', expression_field)

            global_rot_field = GlobalRot(global_rot=global_rot)
            target.add_field('global_rot', global_rot_field)
            body_pose_field = BodyPose(body_pose=body_pose)
            target.add_field('body_pose', body_pose_field)
            hand_pose_field = HandPose(left_hand_pose=left_hand_pose,
                                       right_hand_pose=right_hand_pose)
            target.add_field('hand_pose', hand_pose_field)
            jaw_pose_field = JawPose(jaw_pose=jaw_pose)
            target.add_field('jaw_pose', jaw_pose_field)

            target.add_field(
                'pelvis',
                torch.tensor(self.pelvis[index], dtype=torch.float32))
            target.add_field(
                'translation',
                torch.tensor(self.translation[index], dtype=torch.float32))

        if self.return_gender:
            gender = self.gender[index].decode('utf-8')
            if gender == 'male' or gender == 'female':
                gender = 'M' if gender == 'male' else 'F'
                target.add_field('gender', gender)
            else:
                target.add_field('gender', '')

        if hasattr(self, 'dset_name'):
            dset_name = self.dset_name[index].decode('utf-8')
            vertex_fname = osp.join(
                self.vertex_folder, f'{dset_name}_{index:06d}.npy')
            vertices = np.load(vertex_fname)
            H, W, _ = img.shape

            intrinsics = np.array([[5000, 0, 0.5 * W],
                                   [0, 5000, 0.5 * H],
                                   [0, 0, 1]], dtype=np.float32)

            target.add_field('intrinsics', intrinsics)
            target.add_field('orig_intrinsics', intrinsics)
            vertex_field = Vertices(
                vertices, bc=self.bc, closest_faces=self.closest_faces)
            target.add_field('vertices', vertex_field)

        target.add_field('fname', f'{index:05d}.jpg')

        cropped_image = None
        if self.transforms is not None:
            force_flip = False
            if is_right is not None:
                force_flip = not is_right and self.hand_only
            img, cropped_image, cropped_target = self.transforms(
                img, target, force_flip=force_flip)

        return img, cropped_image, cropped_target, index
