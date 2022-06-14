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


def load_json(
    json_fname: os.PathLike,
    annotations: Dict[str, Any],
    splits_dict: Dict[str, str],
    betas_dict: Optional[Dict[str, Tensor]] = None,
    weights_dict: Optional[Dict[str, Tensor]] = None,
    attributes_dict: Optional[Dict[str, Tensor]] = None,
    identities: Optional[Dict[Tuple[str, str], int]] = None,
    agencies=None,
    split: str = 'train',
):
    ''' Loads the annotations from a json file
    '''

    if agencies is None:
        agencies = []
    with open(json_fname, 'r') as f:
        keypoint_data = json.load(f)

    if len(agencies) < 1:
        agencies = keypoint_data.keys()

    if attributes_dict is not None:
        alist = list(agencies)[0]
        akeys = list(attributes_dict[alist].keys())
        num_attributes = len(attributes_dict[alist][akeys[0]]['attributes'])

    # Remove all the
    to_pop = []
    for key in keypoint_data:
        if key not in agencies:
            to_pop.append(key)
            logger.info(f'Popping: {key}')
    for key in to_pop:
        keypoint_data.pop(key)

    output = defaultdict(lambda: [])
    # Convert and save as numpy arrays
    for agency in tqdm(keypoint_data, desc='Modeling agencies', leave=False):
        num_items = 0
        agency_output = defaultdict(lambda: [])
        for model_name in tqdm(keypoint_data[agency], leave=False):
            #  logger.info(all_keypoints[agency][model_name].keys())

            num_instances = len(keypoint_data[agency][model_name]['images'])
            num_items += num_instances

            if split not in splits_dict[agency][model_name]:
                continue

            index = annotations[agency]['model_name'].index(model_name)
            gender = annotations[agency]['gender'][index]
            height = parse_measurement(annotations[agency]['height_cm'][index])
            chest = parse_measurement(annotations[agency]['bust_cm'][index])
            waist = parse_measurement(annotations[agency]['waist_cm'][index])
            hips = parse_measurement(annotations[agency]['hips_cm'][index])

            agency_output['agency'] += [agency] * num_instances
            agency_output['model_name'] += [model_name] * num_instances
            agency_output['gender'] += [gender] * num_instances
            agency_output['height'] += [height] * num_instances
            agency_output['chest'] += [chest] * num_instances
            agency_output['waist'] += [waist] * num_instances
            agency_output['hips'] += [hips] * num_instances
            agency_output['identity'] += [
                identities[(agency, model_name)]] * num_instances

            #  logger.info(f'{agency}: {model_name}')
            if betas_dict is not None:
                agency_output['betas'] += [
                    betas_dict[agency][model_name]] * num_instances
            if weights_dict is not None:
                agency_output['weight'] += [
                    weights_dict[agency][model_name]] * num_instances
            for key, val in keypoint_data[agency][model_name].items():
                agency_output[key] += val

            # read attribute data
            if attributes_dict is not None:
                if model_name in attributes_dict[agency]:
                    agency_output['attributes'] += [
                        attributes_dict[agency][model_name]['attributes']] * \
                        num_instances
                    agency_output['guess_weight'] += [
                        attributes_dict[agency][model_name]['guess_weight']] * \
                        num_instances
                    agency_output['guess_height'] += [
                        attributes_dict[agency][model_name]['guess_height']] * \
                        num_instances
                    agency_output['guess_age'] += [
                        attributes_dict[agency][model_name]['guess_age']] * \
                        num_instances
                    agency_output['has_attributes'] += [1] * num_instances
                else:
                    agency_output['attributes'] += [[-1] * num_attributes] * \
                        num_instances
                    agency_output['guess_weight'] += [-1] * num_instances
                    agency_output['guess_height'] += [-1] * num_instances
                    agency_output['guess_age'] += [-1] * num_instances
                    agency_output['has_attributes'] += [0] * num_instances

        #  logger.info(f'Output: {num_items}')
        for key, val in agency_output.items():
            output[key] += val

    for key in output:
        output[key] = np.stack(output[key])
        if output[key].dtype == np.float64:
            output[key] = output[key].astype(np.float32)
        #  logger.info(f'{key}: {output[key].shape}')

    return output


def parse_measurement(measurement):
    if type(measurement) is float:
        return measurement / 100
    else:
        return (
            float(measurement.replace(',', '.')) / 100 if measurement else -1)


def load_npz(npz_fname, age):
    keypoint_data = np.load(npz_fname)

    return keypoint_data


class ModelAgency(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(
        self,
        data_folder: os.PathLike,
        img_folder: os.PathLike = 'images',
        annot_fname: os.PathLike = 'cleaned_model_data.json',
        keypoint_fname: os.PathLike = 'keypoints.npz',
        weight_fname: os.PathLike = 'weights.json',
        splits_fname: os.PathLike = 'splits.json',
        identity_fname: os.PathLike = 'final_identities.pkl',
        betas_fname: os.PathLike = 'betas.json',
        attributes_fname: os.PathLike = 'attributes.json',
        param_folder: os.PathLike = 'data/parameters',
        agencies: Optional[List] = None,
        dtype=torch.float32,
        openpose_format: str = 'coco25',
        transforms: Optional[Callable] = None,
        return_params: bool = True,
        body_thresh: float = 0.1,
        hand_thresh: float = 0.2,
        face_thresh: float = 0.4,
        binarization: bool = False,
        keep_only_with_reg=False,
        num_face_keypoints: int = 8,
        use_face_contour: bool = True,
        split: str = 'train',
        metrics: Tuple[str] = ('measurements',),
        return_mass: bool = False,
        only_data_with_attributes: bool = False,
        vertex_flip_correspondences: str = '',
        **kwargs
    ):
        super(ModelAgency, self).__init__()

        data_folder = osp.expanduser(osp.expandvars(data_folder))
        self.data_folder = data_folder
        self.split = split

        msg = [
            'Creating Modelling agency dataset:',
            f'Data folder: {data_folder}',
            f'Agencies: {agencies}',
            f'Body thresh: {body_thresh}',
            f'Binarization: {binarization}',
            f'Metrics: {metrics}',
            f'Return mass: {return_mass}',
        ]
        logger.info('\n'.join(msg))
        self.binarization = binarization

        self.metrics = metrics
        self.transforms = transforms
        self.dtype = dtype
        # Minimum number of keypoints to consider the face a valid point
        self.num_face_keypoints = num_face_keypoints
        self.return_mass = return_mass

        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh

        self.keep_only_with_reg = keep_only_with_reg
        self.openpose_format = openpose_format

        self.use_face_contour = use_face_contour
        self.keyp_format = openpose_format

        self.img_folder = img_folder

        # Folder where the estimated parameters are stored
        self.param_folder = osp.expandvars(param_folder)
        # Flag that decides whether to return the fitted parameters
        self.return_params = return_params
        if self.return_params:
            logger.info(
                f'Loading estimated parameters from: {self.param_folder}')

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

        annot_path = osp.join(data_folder, annot_fname)
        assert osp.exists(annot_path), (
            f'Could not find annotation path: {annot_path}')
        logger.info(f'Loading annotations from: {annot_path}')
        with open(annot_path, 'r') as f:
            annotations = json.load(f)

        splits_path = osp.join(data_folder, splits_fname)
        logger.info(f'Loading splits from: {splits_path}')
        with open(splits_path, 'r') as f:
            splits_dict = json.load(f)

        betas_dict = None
        betas_path = osp.join(data_folder, betas_fname)
        if osp.exists(betas_path):
            logger.info(f'Loading betas from: {betas_path}')
            with open(betas_path, 'r') as f:
                betas_dict = json.load(f)

        weights_dict = None
        weights_path = osp.join(data_folder, weight_fname)
        if osp.exists(weights_path):
            logger.info(f'Loading weights from: {weights_path}')
            with open(weights_path, 'r') as f:
                weights_dict = json.load(f)

        keypoint_path = osp.join(data_folder, keypoint_fname)
        logger.info(f'Loading keypoints from: {keypoint_path}')
        assert osp.exists(keypoint_path), (
            f'Could not find keypoint path: {keypoint_path}')

        identity_path = osp.join(data_folder, identity_fname)
        with open(identity_path, 'rb') as f:
            identity_data = pickle.load(f)

        identities = identity_data['identities']
        neighbors = identity_data['neighbors']

        attributes_path = osp.join(data_folder, attributes_fname)
        if osp.exists(attributes_path):
            logger.info(f'Loading attributes from: {attributes_path}')
            with open(attributes_path, 'r') as f:
                attributes_dict = json.load(f)

        start = time.perf_counter()
        if keypoint_path.endswith('.json'):
            output = load_json(keypoint_path,
                               annotations=annotations,
                               splits_dict=splits_dict,
                               betas_dict=betas_dict,
                               weights_dict=weights_dict,
                               attributes_dict=attributes_dict,
                               agencies=agencies,
                               split=split,
                               identities=identities,
                               )
        elif keypoint_path.endswith('.npz'):
            raise NotImplementedError
        elapsed = time.perf_counter() - start

        logger.info(f'Loading keypoints took: {elapsed:.2f}')

        # use all images in validation set
        #if split == 'train':
        #    attr_mask = output['has_attributes'] if only_data_with_attributes \
        #        else np.ones(len(output['agency']))
        #else:
        #    attr_mask = np.ones(len(output['agency']))

        # support training and testing on attribute subset of model agency data
        # validation is always done on attribute subset.
        if split == 'train' or split == 'test':
            attr_mask = output['has_attributes'] if only_data_with_attributes \
                    else np.ones(len(output['agency']))
        else:
            attr_mask = output['has_attributes']
        attr_mask = np.where(attr_mask == 1)[0]

        self.agencies = output['agency'][attr_mask]
        self.img_fnames = output['images'][attr_mask]
        self._gender = output['gender'][attr_mask]
        self.height = output['height'][attr_mask]
        #self.weights = output['weight'][attr_mask]
        self.chest = output['chest'][attr_mask]
        self.waist = output['waist'][attr_mask]
        self.hips = output['hips'][attr_mask]
        self.model_names = output['model_name'][attr_mask]
        self.body_keypoints = output['body_keypoints'][attr_mask]
        self.left_hand_keypoints = output['left_hand_keypoints'][attr_mask]
        self.right_hand_keypoints = output['right_hand_keypoints'][attr_mask]
        self.face_keypoints = output['face_keypoints'][:, :-2][attr_mask]
        self.identities = output['identity'][attr_mask]
        self.attributes = output['attributes'][attr_mask]
        self.guess_weight = output['guess_weight'][attr_mask]
        self.guess_height = output['guess_height'][attr_mask]
        self.guess_age = output['guess_age'][attr_mask]

        self.num_items = len(self.height)
        logger.info(f'Total number of instances: {self.num_items}')
        #  self.identities = np.unique(self.identities)
        self.num_identities = len(np.unique(self.identities))
        logger.info(f'Total number of identities: {self.num_identities}')

        self.source = 'openpose25_v1'
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
        return 'ModelAgency( \n\t Split: {}\n)'.format(self.split)

    def name(self):
        return 'ModelAgencies/{}'.format(self.split)

    def __len__(self):
        return self.num_items

    @property
    def bmi(self):
        raise NotImplementedError

    @property
    def weight(self):
        return NotImplementedError #self.weights

    @property
    def gender(self):
        return self._gender

    def get_joint_format(self):
        return self.openpose_format

    def only_2d(self):
        # TODO: Support returning params from optimization
        return False

    def get_elements_per_index(self):
        # TODO: Support sampling multiple images of the same identity
        return 1

    def get_keyp_format(self):
        return 'openpose'

    def __getitem__(self, index):
        fname = self.img_fnames[index]
        agency = self.agencies[index]
        model_name = self.model_names[index]

        # Read the image
        img_path = osp.join(
            self.data_folder, agency, self.img_folder, model_name, fname)
        img = read_img(img_path)

        body_keypoints = self.body_keypoints[index]
        left_hand_keypoints = self.left_hand_keypoints[index]
        right_hand_keypoints = self.right_hand_keypoints[index]
        face_keypoints = self.face_keypoints[index]

        keypoints2d = np.concatenate([
            body_keypoints, left_hand_keypoints, right_hand_keypoints,
            face_keypoints], axis=0)

        # Threshold and keep the relevant parts
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
        keypoints = keypoints2d[:, :-1]
        conf = keypoints2d[:, -1]

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

        identity = self.identities[index]
        target.add_field('identity', identity)

        
        orig_bbox = keyps_to_bbox(keypoints, conf, img_size=img.shape)
        orig_center, _, orig_bbox_size = bbox_to_center_scale(
            orig_bbox, dset_scale_factor=dset_scale_factor,
        )
        target.add_field('orig_bbox', orig_bbox)
        target.add_field('orig_center', orig_center)
        target.add_field('orig_bbox_size', orig_bbox_size)
        keypoints_hd = Keypoints2D(
            keypoints2d, img.shape, flip_indices=self.flip_indices,
            flip_axis=0, source=self.source,
            apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        if self.return_mass:
            target.add_field('mass', self.weights[index])
        target.add_field('gender', self._gender[index])

        target.add_field('height', self.height[index])
        #target.add_field('weight', self.weights[index])
        target.add_field('chest', self.chest[index])
        target.add_field('waist', self.waist[index])
        target.add_field('hips', self.hips[index])

        # add attributes
        if self.attributes[index][0] != -1:
            target.add_field('attributes', self.attributes[index])
            #target.add_field('age', self.age[index])

        param_path = osp.join(
            self.param_folder, agency, model_name, fname, 'params.npz')
        if self.return_params and osp.exists(param_path):
            params = np.load(param_path)
            #  logger.info(list(params.keys()))

            focal_length = params['fx']
            camera_center = params['center']

            intrinsics = np.eye(3, dtype=np.float32)

            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            intrinsics[:2, 2] = camera_center

            target.add_field('orig_intrinsics', intrinsics)
            target.add_field('intrinsics', intrinsics)

            translation = params['translation'].squeeze()
            vertices = params['vertices'].astype(np.float32) + translation

            joints = params['joints'].squeeze()
            target.add_field('translation',
                             torch.tensor(translation, dtype=torch.float32))
            # Add the root joint, which we use to correctly change the
            # translation when rotating the mesh
            target.add_field('pelvis', joints[0] - translation)

            vertices_field = Vertices(
                vertices.reshape(-1, 3),
                bc=self.bc, closest_faces=self.closest_faces)
            target.add_field('vertices', vertices_field)

            # Add the global rotation of the body
            global_rot = batch_rot2aa(
                torch.from_numpy(params['global_rot']).reshape(-1, 3, 3))
            global_rot = global_rot.numpy().reshape(-1)
            target.add_field('global_rot', GlobalRot(global_rot))

            # Add the articulation of the body
            body_pose = batch_rot2aa(
                torch.from_numpy(params['body_pose']).reshape(-1, 3, 3))
            body_pose = body_pose.numpy().reshape(-1)
            target.add_field('body_pose', BodyPose(body_pose))

            # Add the pose of the jaw
            jaw_pose = batch_rot2aa(
                torch.from_numpy(params['jaw_pose']).reshape(-1, 3, 3))
            jaw_pose = jaw_pose.numpy().reshape(-1)
            target.add_field('jaw_pose', JawPose(jaw_pose))

            # Add the articulation of the hands
            left_hand_pose = batch_rot2aa(
                torch.from_numpy(params['left_hand_pose']).reshape(-1, 3, 3))
            right_hand_pose = batch_rot2aa(
                torch.from_numpy(params['right_hand_pose']).reshape(-1, 3, 3))
            left_hand_pose = left_hand_pose.numpy().reshape(-1)
            right_hand_pose = right_hand_pose.numpy().reshape(-1)
            target.add_field('hand_pose', HandPose(
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose))

            # Add the shape of the body
            target.add_field('betas', Betas(params['betas']))
            # Add the expression of the face
            target.add_field('expression', Expression(params['expression']))

        #  start = time.perf_counter()
        if self.transforms is not None:
            img, cropped_image, target = self.transforms(img, target)

        target.add_field('fname', osp.join(agency, model_name, fname))
        target.add_field('img_path', img_path)
        #  logger.info('Transforms: {}'.format(time.perf_counter() - start))

        return img, cropped_image, target, index
