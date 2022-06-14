# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: vassilis.choutas@tuebingen.mpg.de
# Contact: ps-license@tuebingen.mpg.de

from typing import Optional, List, Dict, Any, Tuple, Callable
import sys
import os
import os.path as osp
import time
import pickle

import json

from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm

from ..structures import (Keypoints2D, BodyPose, GlobalRot, Betas, Vertices,
                          HandPose, JawPose, Expression, BoundingBox)
from ..utils import (
    get_part_idxs,
    create_flip_indices,
    keyps_to_bbox, bbox_to_center_scale,
    threshold_and_keep_parts,
    KEYPOINT_NAMES_DICT, KEYPOINT_PARTS,
)

from human_shape.utils import read_img, Tensor, batch_rot2aa


class SSP3D(Dataset):

    def __init__(
        self,
        data_folder: os.PathLike,
        img_folder: os.PathLike = 'images',
        silh_folder: os.PathLike = 'silhouettes',
        label_fname: os.PathLike = 'labels_with_vertices.npz',
        dtype=torch.float32,
        openpose_format: str = 'coco25',
        transforms: Optional[Callable] = None,
        return_params: bool = True,
        body_thresh: float = 0.1,
        hand_thresh: float = 0.2,
        face_thresh: float = 0.4,
        binarization: bool = False,
        keep_only_with_reg=False,
        split: str = 'test',
        metrics: Tuple[str] = ('v2v', 'v2v_t'),
        vertex_flip_correspondences: str = '',
        **kwargs
    ):
        super(SSP3D, self).__init__()

        assert 'test' in split, 'SSP3D is a test-only dataset'

        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh
        self.binarization = binarization
        self.dtype = dtype
        self.transforms = transforms

        self.split = split
        self.metrics = metrics
        # If we are loading parameters, then make sure that we load the
        # proper transformation to flip vertices
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

        self.data_folder = osp.expandvars(data_folder)
        logger.info(f'Loading SSP-3D from: {self.data_folder}')

        logger.info(f'Loading SSP-3D labels from: {label_fname}')
        label_fname = osp.expandvars(label_fname)

        self.img_folder = osp.join(self.data_folder, img_folder)
        self.silh_folder = osp.join(self.data_folder, silh_folder)

        # Load the labels
        labels = np.load(label_fname)

        #  ['fnames', 'shapes', 'poses', 'joints2D', 'cam_trans', 'genders',
        #  'bbox_centres', 'bbox_whs']

        self.vertices = labels['vertices']
        self.v_shaped = labels['v_shaped']
        self.fnames = np.asarray(labels['fnames'])
        self.genders = np.asarray(labels['genders'])
        self.bbox_centers = np.asarray(labels['bbox_centres'])
        self.bbox_whs = np.asarray(labels['bbox_whs'])

        self.joints2D = np.asarray(labels['joints2D'])
        logger.info(self.joints2D.shape)
        self.shapes = np.asarray(labels['shapes'])
        self.poses = np.asarray(labels['poses'])

        if 'cam_trans' in labels:
            self.transls = labels['cam_trans']
        elif 'transl' in labels:
            self.transls = labels['transl']

        self.num_items = len(self.vertices)

        self.source = 'coco'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source]
        self.flip_indices = create_flip_indices(self.keypoint_names)

        idxs_dict = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']

        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)
        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)

        self.body_dset_factor = 1.2
        self.head_dset_factor = 2.0
        self.hand_dset_factor = 2.0

    def __repr__(self):
        return f'SSP-3D( \n\t Split: {self.split}\n)'

    def name(self):
        return f'SSP-3D/{self.split}'

    def __len__(self) -> int:
        return self.num_items

    def only_2d(self):
        return False

    def get_elements_per_index(self):
        return 1

    def __getitem__(self, index):

        img_path = osp.join(self.img_folder, self.fnames[index])
        img = read_img(img_path)

        silhouette_fname = osp.join(self.silh_folder, self.fnames[index])
        keypoints2d = self.joints2D[index]
        keypoints2d = threshold_and_keep_parts(
            keypoints2d, self.body_idxs, self.left_hand_idxs,
            self.right_hand_idxs, self.face_idxs,
            body_thresh=self.body_thresh,
            hand_thresh=self.hand_thresh,
            face_thresh=self.face_thresh,
            binarization=self.binarization,
        )

        target = Keypoints2D(
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            flip_axis=0, source=self.source, dtype=self.dtype)

        wh = self.bbox_whs[index]
        center = self.bbox_centers[index]

        xmin, ymin = center - 0.5 * wh
        xmax, ymax = center + 0.5 * wh

        bbox = np.asarray([xmin, ymin, xmax, ymax])
        center, scale, bbox_size = bbox_to_center_scale(
            bbox, dset_scale_factor=self.body_dset_factor)

        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('orig_center', center)
        target.add_field('orig_bbox_size', bbox_size)

        keypoints_hd = Keypoints2D(
            keypoints2d, img.shape, flip_indices=self.flip_indices,
            flip_axis=0, source=self.source,
            apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        vertices = self.vertices[index]

        vertices_field = Vertices(
            vertices.reshape(-1, 3),
            bc=self.bc, closest_faces=self.closest_faces)
        target.add_field('vertices', vertices_field)
        v_shaped = self.v_shaped[index]

        v_shaped_field = Vertices(
            v_shaped.reshape(-1, 3),
            bc=self.bc, closest_faces=self.closest_faces)
        target.add_field('v_shaped', v_shaped_field)

        intrinsics = np.eye(3)
        intrinsics[0, 0] = 5000
        intrinsics[1, 1] = 5000
        intrinsics[:2, 2] = 256
        target.add_field('intrinsics', intrinsics)
        target.add_field('orig_intrinsics', intrinsics)

        #  start = time.perf_counter()
        if self.transforms is not None:
            img, cropped_image, target = self.transforms(img, target)

        name, _ = osp.splitext(self.fnames[index])
        target.add_field('fname', name)
        #  logger.info('Transforms: {}'.format(time.perf_counter() - start))

        return img, cropped_image, target, index
