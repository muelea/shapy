from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle
import yaml

import tqdm
import time
import smplx
import trimesh

import torch
import torch.utils.data as dutils
import numpy as np
import PIL.Image as pil_img

from loguru import logger

#from ..targets import Keypoints2D, Filename, Vertices
#from ..targets.keypoints import dset_to_body_model
from ..structures import (Keypoints2D, BodyPose, GlobalRot, Betas, Vertices,
                          HandPose, JawPose, Expression, BoundingBox)
from ..utils import (
    get_part_idxs,
    create_flip_indices,
    keyps_to_bbox, bbox_to_center_scale,
    threshold_and_keep_parts,
    KEYPOINT_NAMES_DICT, KEYPOINT_PARTS,
)

#from ..utils.bbox import keyps_to_bbox, bbox_to_center_scale

from ..utils import read_keypoints
from ...utils.img_utils import read_img

from body_measurements import BodyMeasurements

FOLDER_MAP_FNAME = 'folder_map.pkl'


class HumanBodyInTheWild(dutils.Dataset):
    """
        If mesh_folder contains smplx registrations, also use the smplx version
        of meas_vertices_path. Even if your regressor predicts smpl meshes. The
        measurements of the predictions will be computed using the smpl measurement 
        tool.
    """
    def __init__(self,
                 data_folder='data/HumanBodyInTheWild',
                 img_folder='photos',
                 keyp_folder='keypoints',
                 imgs_minimal='photos_fullbody_minimal',
                 keyps_minimal='photos_fullbody_minimal_keypoints',
                 mesh_folder='v_shaped/smplx',
                 annot_fname='annotations.yaml',
                 gender_fname='genders.yaml',
                 use_face=True, use_hands=True, use_face_contour=False,
                 model_type='smplx',
                 dtype=torch.float32,
                 joints_to_ign=None,
                 metrics=None,
                 transforms=None,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 binarization=True,
                 split='val',
                 keypoint_source='openpose25_v1',
                 meas_definition_path='body_models/measurement_defitions.yaml',
                 meas_vertices_path='body_models/smplx_measurements.yaml',
                 body_model_folder='body_models',
                 skip_multi_person=True,
                 **kwargs):
        super(HumanBodyInTheWild, self).__init__()

        self.device = device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.binarization = binarization

        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh
        self.split = split

        self.data_folder = osp.expandvars(data_folder)
        self.img_folder = osp.join(self.data_folder, img_folder, self.split)
        self.imgs_minimal = osp.join(self.data_folder, imgs_minimal, self.split)
        self.keyps_minimal = osp.join(self.data_folder, keyps_minimal, self.split)
        self.keyp_folder = osp.join(self.data_folder, keyp_folder, self.split)
        self.mesh_folder = osp.join(self.data_folder, mesh_folder, self.split)

        # get the ground truth mesh
        gt_v_shaped = {}
        if self.split in ['val', 'test']:
            for fname in os.listdir(self.mesh_folder):
                if fname.startswith('.') or fname[-4:] != '.obj':
                    continue
                subject_id = osp.splitext(fname)[0]
                path = osp.join(self.mesh_folder, fname)
                mesh = trimesh.load(path, process=False)
                gt_v_shaped[subject_id] = np.array(mesh.vertices)

        gender_path = osp.join(self.data_folder, gender_fname)
        with open(gender_path, 'r') as f:
            gender_data = yaml.safe_load(f)
        
        # load measurement module
        measurements = BodyMeasurements({
            'meas_definition_path': osp.join(meas_definition_path),
            'meas_vertices_path': osp.join(meas_vertices_path),
        }).to(device=self.device)

        # Load SMPL model
        body_model = smplx.create(
            model_path=body_model_folder, 
            model_type=model_type
        )
        body_model_faces = torch.tensor(
            body_model.faces.astype('int64'), dtype=torch.int64)

        self.img_paths = []
        img_fnames = []
        subject_ids = []
        keypoints2d = []
        genders = []
        v_shaped = []
        height = []
        chest = []
        waist = []
        hips = []

        skip_multi_counter = 0
        skipped_img_paths = []
        subject_folders = sorted(os.listdir(self.img_folder))
        for subject_folder in tqdm.tqdm(subject_folders):

            if subject_folder.startswith('.'):
                continue

            tokens = subject_folder.split('_')
            subject_id = tokens[0]

            # add measurements
            if self.split in ['val', 'test']:
                curr_v_shaped = torch.from_numpy(gt_v_shaped[subject_id]).unsqueeze(0).float()
                curr_triangles = curr_v_shaped[:, body_model_faces].to(self.device)
                curr_subj_meas_val = measurements(curr_triangles)['measurements']

            curr_subj_path = osp.join(self.img_folder, subject_folder)

            for img_type in tqdm.tqdm(os.listdir(curr_subj_path), desc='Type'):
                if img_type.startswith('.'):
                    continue
                img_type_path = osp.join(curr_subj_path, img_type)
                keyp_path = osp.join(
                    self.keyp_folder, subject_folder, img_type)

                for img_fname in tqdm.tqdm(os.listdir(img_type_path),
                                           desc='Images'):
                    if img_fname.startswith('.'):
                        continue

                    curr_img_path = osp.join(img_type_path, img_fname)
                    assert osp.exists(curr_img_path), (
                        f'{curr_img_path} not found')

                    fname, _ = osp.splitext(img_fname)
                    curr_keyp_path = osp.join(
                        keyp_path, f'{fname}.json')

                    if not osp.exists(curr_keyp_path):
                        correct_fname = fname.replace(
                            '(', '').replace(')', '').replace(
                            ' ', '_')
                        curr_keyp_path = osp.join(
                            keyp_path,
                            f'{correct_fname}' +
                            '.json')

                    if not osp.exists(curr_keyp_path):
                        logger.warning(f'{curr_keyp_path} not found')
                        continue

                    keyp_data = read_keypoints(curr_keyp_path)
                    if skip_multi_person and len(keyp_data) != 1:
                        skip_multi_counter += 1
                        skipped_img_paths += [curr_img_path]
                        #if len(keyp_data) > 1:
                        #    logger.warning(f'Skipping {curr_img_path}, because of multiple OpenPose detections.')
                        #else:
                        #    logger.warning(f'Skipping {curr_img_path}.')
                        continue

                    self.img_paths.append(curr_img_path)
                    img_fnames.append(img_fname)
                    subject_ids.append(subject_id)
                    genders.append(gender_data[subject_id])
                    keypoints2d.append(keyp_data)
                    if self.split in ['val', 'test']:
                        v_shaped.append(gt_v_shaped[subject_id])
                        height.append(curr_subj_meas_val['height']['tensor'].item())
                        chest.append(curr_subj_meas_val['chest']['tensor'].item())
                        waist.append(curr_subj_meas_val['waist']['tensor'].item())
                        hips.append(curr_subj_meas_val['hips']['tensor'].item())
        
        self.keypoints2d = np.concatenate(keypoints2d, axis=0)
        self.subject_ids = np.asarray(subject_ids)
        self.img_fnames = img_fnames
        self.genders = np.asarray(genders)
        if self.split in ['val', 'test']:
            self.v_shaped = np.stack(v_shaped)
            self.height = np.asarray(height)
            self.chest = np.asarray(chest)
            self.waist = np.asarray(waist)
            self.hips = np.asarray(hips)

        self.num_items = len(self.img_paths)

        self.transforms = transforms
        self.dtype = dtype

        self.use_face = use_face
        self.use_hands = use_hands
        self.use_face_contour = use_face_contour
        self.model_type = model_type

        #source_idxs, target_idxs = dset_to_body_model(
        #    dset='openpose25+hands+face',
        #    model_type='smplx', use_hands=True, use_face=True,
        #    use_face_contour=self.use_face_contour)
        #source_idxs = np.arange(self.keypoints2d[0].shape[0])
        #target_idxs = np.arange(self.keypoints2d[0].shape[0])
        #self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        #self.target_idxs = np.asarray(target_idxs, dtype=np.int64)

        self.source = keypoint_source
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source]
        self.flip_indices = create_flip_indices(self.keypoint_names)

        idxs_dict = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)
        body_idxs = idxs_dict['body']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]

        self.body_idxs = np.asarray(body_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)
        self.face_idxs = np.asarray(face_idxs)

        logger.warning(f'Skipped {skip_multi_counter} images, because of != 1 OpenPose detections.')
        for item in skipped_img_paths:
            print(item)
            
    def name(self):
        return f'HumanBodyInTheWild/{self.split}'

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return f'HumanBodyInTheWild/{self.split}'

    def __len__(self):
        return self.num_items

    def only_2d(self):
        return False

    def get_img_info(self, index):
        img = pil_img.open(self.img_paths[index])
        width, height = img.size
        return {'height': height, 'width': width}

    def __getitem__(self, index):
        img_fn = self.img_paths[index]
       
        img = read_img(img_fn)

        # Pad to compensate for extra keypoints
        #output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
        #                               3], dtype=np.float32)

        keypoints2d = self.keypoints2d[index]

        # Threshold and keep the relevant parts
        keypoints2d = threshold_and_keep_parts(
            keypoints2d, self.body_idxs, self.left_hand_idxs,
            self.right_hand_idxs, self.face_idxs,
            body_thresh=self.body_thresh,
            hand_thresh=self.hand_thresh,
            face_thresh=self.face_thresh,
            binarization=self.binarization,
        )

        '''
        output_keypoints2d[self.target_idxs] = keypoints[self.source_idxs]

        # Remove joints with negative confidence
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0
        if self.body_thresh > 0:
            # Only keep the points with confidence above a threshold
            body_conf = output_keypoints2d[self.body_idxs, -1]
            hand_conf = output_keypoints2d[self.hand_idxs, -1]
            face_conf = output_keypoints2d[self.face_idxs, -1]

            body_conf[body_conf < self.body_thresh] = 0.0
            hand_conf[hand_conf < self.hand_thresh] = 0.0
            face_conf[face_conf < self.face_thresh] = 0.0
            if self.binarization:
                body_conf = (
                    body_conf >= self.body_thresh).astype(
                        output_keypoints2d.dtype)
                hand_conf = (
                    hand_conf >= self.hand_thresh).astype(
                        output_keypoints2d.dtype)
                face_conf = (
                    face_conf >= self.face_thresh).astype(
                        output_keypoints2d.dtype)

            output_keypoints2d[self.body_idxs, -1] = body_conf
            output_keypoints2d[self.hand_idxs, -1] = hand_conf
            output_keypoints2d[self.face_idxs, -1] = face_conf
        '''

        target = Keypoints2D(
            keypoints2d, 
            img.shape, 
            flip_indices=self.flip_indices,
            flip_axis=0, 
            source=self.source, 
            dtype=self.dtype
        )

        target_hd = Keypoints2D(
            keypoints2d, 
            img.shape, 
            flip_indices=self.flip_indices,
            flip_axis=0, 
            source=self.source, 
            apply_crop=False,
            dtype=self.dtype
        )
        target.add_field('keypoints_hd', target_hd)


        keypoints = keypoints2d[:, :-1]
        conf = keypoints2d[:, -1]
        bbox = keyps_to_bbox(keypoints, conf, img_size=img.shape)
        center, scale, bbox_size = bbox_to_center_scale(
            bbox, dset_scale_factor=1.2 )
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('orig_bbox_size', bbox_size)
        target.add_field('orig_bbox', bbox)
        target.add_field('orig_center', center)
        target.add_field('bbox', bbox)

        if self.split in ['val', 'test']:
            target.add_field('v_shaped', Vertices(self.v_shaped[index]))

        target.add_field('gender', self.genders[index])

        #  start = time.perf_counter()
        if self.transforms is not None:
            #img, target = self.transforms(img, target, dset_scale_factor=1.2)
            img, cropped_image, target = self.transforms(img, target, dset_scale_factor=1.2)

        img_only_fn = osp.split(img_fn)[1]
        target.add_field('fname', img_only_fn)
        target.add_field('filename', img_fn)

        if self.split in ['val', 'test']:
            target.add_field('height', self.height[index])
            target.add_field('chest', self.chest[index])
            target.add_field('waist', self.waist[index])
            target.add_field('hips', self.hips[index])

        return img, cropped_image, target, index
