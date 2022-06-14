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
import json

from loguru import logger

from ..structures import Keypoints2D
from ..utils import (
    get_part_idxs,
    create_flip_indices,
    read_keypoints,
    keyps_to_bbox, bbox_to_center_scale,
    KEYPOINT_NAMES_DICT, KEYPOINT_PARTS,
)

from human_shape.utils import read_img, nand, binarize


class OpenPose(dutils.Dataset):
    def __init__(self,
                 data_folder='data/openpose',
                 img_folder='images',
                 keyp_folder='keypoints',
                 split='train',
                 head_only=False,
                 hand_only=False,
                 is_right=True,
                 use_face=True, use_hands=True, use_face_contour=False,
                 dtype=torch.float32,
                 metrics=None,
                 transforms=None,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 body_dset_factor=1.2,
                 hand_dset_factor=2.0,
                 head_dset_factor=2.0,
                 binarization=True,
                 **kwargs):

        super(OpenPose, self).__init__()
        assert nand(head_only, hand_only), (
            'Hand only and head only can\'t be True at the same time')

        self.is_right = is_right
        self.head_only = head_only
        self.hand_only = hand_only
        logger.info(f'Hand only: {self.hand_only}')
        logger.info(f'Is right: {self.is_right}')

        self.body_dset_factor = body_dset_factor
        self.head_dset_factor = head_dset_factor
        self.hand_dset_factor = hand_dset_factor

        self.split = split
        self.is_train = 'train' in split

        self.data_folder = osp.expandvars(osp.expanduser(data_folder))
        self.img_folder = osp.join(self.data_folder, img_folder)
        self.keyp_folder = osp.join(self.data_folder, keyp_folder)

        self.transforms = transforms
        self.dtype = dtype

        self.use_face = use_face
        self.use_hands = use_hands
        self.use_face_contour = use_face_contour
        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh
        self.binarization = binarization

        self.img_paths = []
        self.keypoints = []
        for img_fname in os.listdir(self.img_folder):
            fname, _ = osp.splitext(img_fname)

            keyp_path = osp.join(
                self.keyp_folder, '{}_keypoints.json'.format(fname))
            if not osp.exists(keyp_path):
                keyp_path = osp.join(
                self.keyp_folder, '{}.json'.format(fname))
                if not osp.exists(keyp_path):
                    continue

            keypoints = read_keypoints(keyp_path)
            if keypoints is None:
                continue

            img_path = osp.join(self.img_folder, img_fname)
            self.img_paths += [img_path] * keypoints.shape[0]
            self.keypoints.append(keypoints)

        self.keypoints = np.concatenate(self.keypoints, axis=0)
        self.num_items = len(self.img_paths)

        self.source = 'openpose25_v1'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source]
        self.flip_indices = create_flip_indices(self.keypoint_names)

        idxs_dict = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]

        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)

    def __repr__(self):
        return f'OpenPose( \n\t Split: {self.split}\n)'

    def name(self):
        return 'OpenPose'

    def __len__(self):
        return self.num_items

    def get_elements_per_index(self):
        return 1

    def only_2d(self):
        return True

    def __getitem__(self, index):
        img_fn = self.img_paths[index]
        img = read_img(img_fn)

        keypoints2d = self.keypoints[index].copy()
        keypoints2d[:, -1] = np.clip(keypoints2d[:, -1], 0, 1)

        is_right = self.is_right

        body_conf = keypoints2d[self.body_idxs, -1]
        face_conf = keypoints2d[self.face_idxs, -1]
        left_hand_conf = keypoints2d[self.left_hand_idxs, -1]
        right_hand_conf = keypoints2d[self.right_hand_idxs, -1]
        # Only keep the points with confidence above a threshold
        if self.body_thresh > 0:
            body_conf[body_conf < self.body_thresh] = 0.0
        if self.face_thresh > 0:
            face_conf[face_conf < self.face_thresh] = 0.0

        if self.hand_thresh > 0:
            left_hand_conf[left_hand_conf < self.hand_thresh] = 0.0
            right_hand_conf[right_hand_conf < self.hand_thresh] = 0.0

        if self.head_only or self.hand_only:
            body_conf[:] = 0.0
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
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source,
            dtype=self.dtype)

        keypoints = keypoints2d[:, :-1]
        conf = keypoints2d[:, -1]
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
        if center is None:
            return None, None, None, None
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)

        orig_center, _, orig_bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.shape),
            dset_scale_factor=dset_scale_factor,
        )
        target.add_field('orig_center', orig_center)
        target.add_field('orig_bbox_size', orig_bbox_size)
        keypoints_hd = Keypoints2D(
            keypoints2d, img.shape, flip_indices=self.flip_indices,
            flip_axis=0, source=self.source,
            apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        #  start = time.perf_counter()
        if self.transforms is not None:
            force_flip = not self.is_right and self.hand_only
            img, cropped_image, target = self.transforms(
                img, target, force_flip=force_flip)

        img_fn = osp.split(img_fn)[1]
        target.add_field('fname', img_fn)

        return img, cropped_image, target, index


class OpenPoseTracks(dutils.Dataset):
    def __init__(self, data_folder='data/openpose_tracks',
                 img_folder='images',
                 keyp_folder='keypoints',
                 split='train',
                 head_only=False,
                 hand_only=False,
                 is_right=False,
                 use_face=True, use_hands=True, use_face_contour=False,
                 pid=4,
                 keyp_format='coco25',
                 dtype=torch.float32,
                 metrics=None,
                 transforms=None,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 body_dset_factor=1.2,
                 hand_dset_factor=2.0,
                 head_dset_factor=2.0,
                 binarization=True,
                 limit=1500,
                 **kwargs):

        super(OpenPoseTracks, self).__init__()
        assert nand(head_only, hand_only), (
            'Hand only and head only can\'t be True at the same time')

        self.is_right = is_right
        self.head_only = head_only
        self.hand_only = hand_only
        logger.info(f'Hand only: {self.hand_only}')
        logger.info(f'Is right: {self.is_right}')

        self.body_dset_factor = body_dset_factor
        self.head_dset_factor = head_dset_factor
        self.hand_dset_factor = hand_dset_factor

        self.split = split
        self.is_train = 'train' in split

        self.data_folder = osp.expandvars(osp.expanduser(data_folder))
        self.img_folder = osp.join(self.data_folder, img_folder)
        self.keyp_folder = osp.join(self.data_folder, keyp_folder)

        self.transforms = transforms
        self.dtype = dtype

        self.use_face = use_face
        self.use_hands = use_hands
        self.use_face_contour = use_face_contour
        self.keyp_format = keyp_format
        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh
        self.binarization = binarization

        track_path = osp.join(self.data_folder, 'by_id.json')
        with open(track_path, 'r') as f:
            track_data = json.load(f)[f'{pid}']

        self.num_items = len(track_data)

        logger.info(track_data[0].keys())
        imgnames = []
        keypoints = []
        for idx, d in enumerate(track_data):
            keyps = np.array(d['keypoints'], dtype=np.float32)[:-2]
            keypoints.append(keyps)
            imgnames.append(d['fname'])
        self.keypoints = np.stack(keypoints)
        self.imgnames = np.stack(imgnames)
        if limit > 0:
            self.keypoints = self.keypoints[:-limit]
            self.imgnames = self.imgnames[:-limit]

        self.source = 'openpose25_v1'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source]
        self.flip_indices = create_flip_indices(self.keypoint_names)

        idxs_dict = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]

        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)

        import ipdb;ipdb.set_trace()

    def __repr__(self):
        return 'OpenPose( \n\t Split: {}\n)'.format(self.split)

    def name(self):
        return 'OpenPose'

    def __len__(self):
        return self.num_items

    def get_elements_per_index(self):
        return 1

    def only_2d(self):
        return True

    def __getitem__(self, index):
        img_fn = osp.join(self.img_folder, self.imgnames[index])
        img = read_img(img_fn)

        keypoints2d = self.keypoints[index].copy()
        keypoints2d[:, -1] = np.clip(keypoints2d[:, -1], 0, 1)

        is_right = self.is_right

        # Only keep the points with confidence above a threshold
        body_conf = keypoints2d[self.body_idxs, -1]
        face_conf = keypoints2d[self.face_idxs, -1]
        left_hand_conf = keypoints2d[self.left_hand_idxs, -1]
        right_hand_conf = keypoints2d[self.right_hand_idxs, -1]
        if self.body_thresh > 0:
            body_conf[body_conf < self.body_thresh] = 0.0
        if self.face_thresh > 0:
            face_conf[face_conf < self.face_thresh] = 0.0

        if self.hand_thresh > 0:
            left_hand_conf[left_hand_conf < self.hand_thresh] = 0.0
            right_hand_conf[right_hand_conf < self.hand_thresh] = 0.0

        if self.head_only or self.hand_only:
            body_conf[:] = 0.0
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
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source,
            dtype=self.dtype)

        keypoints = keypoints2d[:, :-1]
        conf = keypoints2d[:, -1]
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
        if center is None:
            return None, None, None, None
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)

        orig_center, _, orig_bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.shape),
            dset_scale_factor=dset_scale_factor,
        )
        target.add_field('orig_center', orig_center)
        target.add_field('orig_bbox_size', orig_bbox_size)

        keypoints_hd = Keypoints2D(
            keypoints2d, img.shape, flip_indices=self.flip_indices,
            flip_axis=0, source=self.source,
            apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        target.add_field('fname', self.imgnames[index])
        if self.transforms is not None:
            force_flip = not self.is_right and self.hand_only
            img, cropped_image, target = self.transforms(
                img, target, force_flip=force_flip)

        return img, cropped_image, target, index
