import sys
import os
import os.path as osp

import pickle

import torch
import torch.utils.data as dutils
import numpy as np
import cv2
import yaml
from loguru import logger

from ..structures import Keypoints2D, Keypoints3D, Vertices, Joints
from ..utils import (
    get_part_idxs,
    create_flip_indices,
    keyps_to_bbox, bbox_to_center_scale,
    KEYPOINT_NAMES_DICT, KEYPOINT_PARTS,
    threshold_and_keep_parts,
)

from human_shape.utils import read_img


class EHF(dutils.Dataset):

    def __init__(self,
                 data_folder,
                 img_folder='images',
                 alignments_folder='alignments',
                 dtype=torch.float32,
                 transforms=None,
                 split='train',
                 keyp_format='coco25',
                 metrics=None,
                 head_only=False,
                 hand_only=False,
                 is_right=True,
                 binarization=True,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 body_dset_factor=1.2,
                 hand_dset_factor=2.0,
                 head_dset_factor=2.0,
                 **kwargs):
        super(EHF, self).__init__()
        if metrics is None:
            metrics = ['v2v']
        self.metrics = metrics

        self.dtype = dtype
        self.data_folder = osp.expandvars(data_folder)
        self.img_folder = img_folder
        self.alignments_folder = alignments_folder

        keypoint_fname = osp.join(self.data_folder, 'gt_keyps.npz')
        keypoint_data = np.load(keypoint_fname)
        self.keypoints = keypoint_data['gt_keypoints_2d'].astype(np.float32)
        self.keypoints3d = keypoint_data['gt_keypoints_3d'].astype(np.float32)
        self.joints14 = keypoint_data['gt_joints14'].astype(np.float32)

        self.is_train = 'train' in split
        self.split = split
        self.keyp_format = keyp_format
        self.is_right = is_right
        self.head_only = head_only
        self.hand_only = hand_only
        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh
        self.binarization = binarization

        self.body_dset_factor = body_dset_factor
        self.head_dset_factor = head_dset_factor
        self.hand_dset_factor = hand_dset_factor

        annot_fn = osp.join(self.data_folder, 'annotations.yaml')
        with open(annot_fn, 'r') as annot_file:
            annotations = yaml.load(annot_file)
        self.annotations = annotations
        self.annotations = (self.annotations['train'] +
                            self.annotations['test'])

        self.transforms = transforms

        self.img_fns = sorted(
            os.listdir(osp.join(self.data_folder, self.img_folder)))

        self.source = 'ehf'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source]
        self.flip_indices = create_flip_indices(self.keypoint_names)

        idxs_dict = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']

        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)

    def __repr__(self):
        return 'EHF'

    def name(self):
        return 'EHF/Test'

    def get_num_joints(self):
        return 14

    def __len__(self):
        return len(self.img_fns)

    def get_elements_per_index(self):
        return 1

    def __getitem__(self, index):
        fn = self.annotations[index]
        img_path = osp.join(self.data_folder, self.img_folder, f'{fn}.png')
        img = read_img(img_path)

        _, fn = os.path.split(fn)

        # Copy keypoints from the GT data
        is_right = self.is_right
        # Remove joints with negative confidence

        keypoints2d = self.keypoints[index].copy()
        keypoints2d = np.concatenate(
            [keypoints2d, np.ones_like(keypoints2d[:, [0]])], axis=-1)
        keypoints3d = self.keypoints3d[index].copy()
        keypoints3d = np.concatenate(
            [keypoints3d, np.ones_like(keypoints3d[:, [0]])], axis=-1)

        keypoints2d = threshold_and_keep_parts(
            keypoints2d,
            body_idxs=self.body_idxs,
            left_hand_idxs=self.left_hand_idxs,
            right_hand_idxs=self.right_hand_idxs,
            face_idxs=self.face_idxs,
            body_thresh=self.body_thresh,
            hand_thresh=self.hand_thresh,
            face_thresh=self.face_thresh,
            hand_only=self.hand_only,
            head_only=self.head_only,
            binarization=self.binarization,
        )

        keypoints3d = threshold_and_keep_parts(
            keypoints3d,
            body_idxs=self.body_idxs,
            left_hand_idxs=self.left_hand_idxs,
            right_hand_idxs=self.right_hand_idxs,
            face_idxs=self.face_idxs,
            body_thresh=self.body_thresh,
            hand_thresh=self.hand_thresh,
            face_thresh=self.face_thresh,
            hand_only=self.hand_only,
            head_only=self.head_only,
            binarization=self.binarization,
        )

        target = Keypoints2D(
            keypoints2d,
            img.shape, flip_indices=self.flip_indices,
            flip_axis=0, source=self.source, dtype=self.dtype)

        keypoints = keypoints2d[:, :-1]
        conf = keypoints2d[:, -1]
        if self.head_only:
            dset_scale_factor = self.head_dset_factor
        elif self.hand_only:
            dset_scale_factor = self.hand_dset_factor
        else:
            dset_scale_factor = self.body_dset_factor

        center, scale, bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.size),
            dset_scale_factor=dset_scale_factor,
        )
        if center is None:
            return None, None, None, None

        if self.hand_only:
            target.add_field('is_right', is_right)
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)

        keypoints_hd = Keypoints2D(
            keypoints2d,
            img.shape, flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source, apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        target.add_field(
            'keypoints3d',
            Keypoints3D(keypoints3d, img.shape, flip_axis=0,
                        source=self.source, flip_indices=self.flip_indices)
        )

        body_box = keyps_to_bbox(keypoints, conf, img_size=img.size)

        head_box = keyps_to_bbox(
            keypoints[self.face_idxs], conf[self.face_idxs],
            scale=self.head_dset_factor,
            img_size=img.size)
        left_hand_box = keyps_to_bbox(
            keypoints[self.left_hand_idxs], conf[self.left_hand_idxs],
            scale=self.hand_dset_factor,
            img_size=img.size)
        right_hand_box = keyps_to_bbox(
            keypoints[self.right_hand_idxs], conf[self.right_hand_idxs],
            scale=self.hand_dset_factor,
            img_size=img.size
        )

        target.add_field('body_box', body_box)
        target.add_field('left_hand_box', left_hand_box)
        target.add_field('right_hand_box', right_hand_box)
        target.add_field('head_box', head_box)

        orig_center, _, orig_bbox_size = bbox_to_center_scale(
            body_box, dset_scale_factor=1.0,
        )
        target.add_field('orig_center', orig_center)
        target.add_field('orig_bbox_size', bbox_size)

        alignment_path = osp.join(self.data_folder, self.alignments_folder,
                                  fn.replace('.07_C', '') + '.pkl')
        with open(alignment_path, 'rb') as alignment_file:
            alignment_data = pickle.load(alignment_file, encoding='latin1')

        transl = np.array([-0.03609917, 0.43416458, 2.37101226])
        camera_pose = np.array([-2.9874789618512025, 0.011724572107320893,
                                -0.05704686818955933])
        camera_pose = cv2.Rodrigues(camera_pose)[0]

        vertices = alignment_data['v']
        cam_vertices = vertices.dot(camera_pose.T) + transl.reshape(1, 3)

        vertices_field = Vertices(cam_vertices)
        target.add_field('vertices', vertices_field)

        H, W, _ = img.shape
        intrinsics = np.array([[1498.22426237, 0, 790.263706],
                               [0, 1498.22426237, 578.90334],
                               [0, 0, 1]], dtype=np.float32)
        target.add_field('intrinsics', intrinsics)

        joints3d = self.joints14[index]
        cam_joints14 = joints3d.dot(camera_pose.T) + transl.reshape(1, 3)
        joints = Joints(cam_joints14[:14])
        target.add_field('joints14', joints)

        if self.transforms is not None:
            force_flip = False
            if self.hand_only and not is_right:
                force_flip = True
            img, cropped_image, target = self.transforms(
                img, target, force_flip=force_flip)

        target.add_field('fname', fn)
        return img, cropped_image, target, index
