from typing import List, Tuple, Dict, Optional

import sys
import os
import os.path as osp
from copy import deepcopy

import pickle

import time

import numpy as np
from loguru import logger
from collections import defaultdict
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lbs import (
    lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords, blend_shapes)
from .utils import (
    find_joint_kin_chain, to_tensor, JointsFromVerticesSelector,
    KeypointTensor,
)

from human_shape.data.utils import (
    KEYPOINT_NAMES_DICT, KEYPOINT_CONNECTIONS,
    KEYPOINT_PARTS,
    PART_NAMES,
    get_part_idxs,
    kp_connections)
from human_shape.utils import to_np, Struct, StringList, IntList, Tensor

J14_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head',
]

J9_NAMES = [
    'right_hip',
    'left_hip',
    'neck',
    'top of head ',
    'pelvis',
    'thorax ',
    'spine',
    'jaw ',
    'head',
]


class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300
    NAME = 'smpl'

    def __init__(self,
                 model_folder='data/models/smpl',
                 data_struct=None,
                 betas=None,
                 gender='neutral',
                 vertex_ids=None,
                 ext='npz',
                 extra_joint_path='',
                 v_template_path='',
                 dtype=torch.float32,
                 head_verts_ids_path: str = '',
                 **kwargs):
        '''
            Keyword Arguments:
                -
        '''

        if data_struct is None:
            model_fn = f'SMPL_{gender.upper()}.{ext}'
            smpl_path = os.path.join(model_folder, model_fn)
            if ext == 'npz':
                model_data = np.load(smpl_path, allow_pickle=True)
            else:
                with open(smpl_path, 'rb') as smpl_file:
                    model_data = pickle.load(smpl_file, encoding='latin1')
            data_struct = Struct(**model_data)

        super(SMPL, self).__init__()
        self.gender = gender
        if betas is None:
            betas = {'num': 10}
        self._num_betas = betas.get('num', 10)

        self.dtype = dtype

        if extra_joint_path:
            self.extra_joint_selector = JointsFromVerticesSelector(
                fname=extra_joint_path, **kwargs)

        self.faces = to_np(data_struct.f, dtype=np.int64)
        self.register_buffer(
            'faces_tensor', to_tensor(self.faces, dtype=torch.long))

        # The vertices of the template model
        v_template_path = osp.expandvars(v_template_path)
        if osp.exists(v_template_path):
            logger.info(f'Loading v template from: {v_template_path}')
            v_template_mesh = trimesh.load_mesh(
                v_template_path, process=False)
            v_template = np.asarray(v_template_mesh.vertices)
            self.register_buffer(
                'v_template', to_tensor(v_template, dtype=dtype))
        else:
            # Use the default vertex template stored in the model
            v_template = to_tensor(to_np(data_struct.v_template), dtype=dtype)
            self.register_buffer('v_template', v_template)

        # The path the vertex indices of the head
        head_verts_ids_path = osp.expandvars(head_verts_ids_path)
        if osp.exists(head_verts_ids_path):
            head_vertices_ids = np.load(head_verts_ids_path)
        else:
            head_vertices_ids = []
        head_vertices_ids = torch.tensor(head_vertices_ids, dtype=torch.long)
        self.register_buffer('head_vertices_ids', head_vertices_ids)

        # Sanity check to ensure that the betas don't go in the expression
        # space
        num_betas = min(self.num_betas, self.SHAPE_SPACE_DIM)
        # The shape components
        shapedirs = data_struct.shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer('shapedirs',
                             to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer(
            'lbs_weights',
            to_tensor(to_np(data_struct.weights), dtype=dtype))

        self._keypoint_names = self.build_keypoint_names()

        j14_regressor_path = osp.expandvars(
            kwargs.get('j14_regressor_path', ''))
        self.use_joint_regressor = osp.exists(j14_regressor_path)
        logger.info(f'Use joint regressor: {self.use_joint_regressor}')
        if self.use_joint_regressor:
            logger.info(
                f'Loading joint regressor from: {j14_regressor_path}')
            if j14_regressor_path.endswith('.pkl'):
                with open(j14_regressor_path, 'rb') as f:
                    j14_regressor = pickle.load(f, encoding='latin1')
            elif j14_regressor_path.endswith('.npy'):
                j14_regressor = np.load(j14_regressor_path)

            if j14_regressor.shape[0] == 14:
                target_names = J14_NAMES
            elif j14_regressor.shape[0] == 9:
                target_names = J9_NAMES

            source = []
            target = []
            for idx, name in enumerate(self.keypoint_names):
                if name in target_names:
                    source.append(idx)
                    target.append(target_names.index(name))
            source = np.asarray(source)
            target = np.asarray(target)
            self.register_buffer('source_idxs', torch.from_numpy(source))
            self.register_buffer('target_idxs', torch.from_numpy(target))

            extra_joint_regressor = torch.from_numpy(
                j14_regressor).to(dtype=torch.float32)
            self.register_buffer(
                'extra_joint_regressor', extra_joint_regressor)

    @property
    def name(self):
        return self.NAME

    def get_head_vertices_ids(self):
        return self.head_vertices_ids

    @property
    def num_body_joints(self):
        return self.NUM_BODY_JOINTS

    @property
    def num_joints(self) -> int:
        return self.J_regressor.shape[0]

    @property
    def num_verts(self) -> int:
        ''' Returns the number of vertices in the MANO mesh '''
        return self.v_template.shape[0]

    @property
    def keypoint_names(self) -> StringList:
        return self._keypoint_names

    @property
    def connections(self) -> List[Tuple[int, int]]:
        if not hasattr(self, '_connections'):
            connections = kp_connections(
                self.keypoint_names, KEYPOINT_CONNECTIONS)
            self._connections = connections

        return self._connections

    @property
    def parts(self) -> Dict[str, Dict[str, IntList]]:
        if not hasattr(self, '_parts'):
            parts = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)
            self._parts = parts
        return self._parts

    @property
    def part_connections(self) -> List[Tuple[int, int]]:
        if not hasattr(self, '_part_connections'):
            _part_connections = {}
            for part_name in PART_NAMES:
                _part_connections[part_name] = kp_connections(
                    self.keypoint_names, KEYPOINT_CONNECTIONS,
                    part=part_name, keypoint_parts=KEYPOINT_PARTS)
            self._part_connections = _part_connections

        return self._part_connections

    def build_keypoint_names(self) -> StringList:
        model_keypoint_names = KEYPOINT_NAMES_DICT[self.NAME.lower()]
        if hasattr(self, 'extra_joint_selector'):
            model_keypoint_names += (
                self.extra_joint_selector.extra_joint_names())
        return model_keypoint_names

    def extra_repr(self) -> str:
        msg = [
            f'Gender: {self.gender.upper()}',
            f'Number of joints: {self.J_regressor.shape[0]}',
            f'Betas: {self.num_betas}',
        ]
        return '\n'.join(msg)

    @property
    def num_betas(self):
        return self._num_betas

    @torch.no_grad()
    def reset_params(self, **params_dict):
        # TODO: In PyTorch 1.0 just set recurse=False
        for param_name, param in self.named_parameters():
            if 'decoder' in param_name:
                continue
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def forward_shape(
        self,
        betas: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        betas = betas if betas is not None else self.betas
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        return {
            'vertices': v_shaped,
            'betas': betas,
            'v_shaped': v_shaped,
        }

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_rot: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_shaped: bool = True,
        get_skin: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> Dict[str, Tensor]:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            get_skin: bool, optional
                Return the vertices and the joints of the model. (default=true)
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        device = self.shapedirs.device
        dtype = self.shapedirs.dtype
        model_vars = [betas, global_rot, body_pose, transl]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_rot is None:
            global_rot = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(
                    batch_size, self.NUM_BODY_JOINTS, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)

        full_pose = torch.cat([global_rot, body_pose], dim=1)

        lbs_output = lbs(betas, full_pose, self.v_template,
                         self.shapedirs, self.posedirs,
                         self.J_regressor, self.parents,
                         self.lbs_weights,
                         pose2rot=False, return_shaped=return_shaped)
        vertices = lbs_output['vertices']
        joints = lbs_output['joints']

        final_joint_set = [joints]
        if hasattr(self, 'extra_joint_selector'):
            # Add any extra joints that might be needed
            extra_joints = self.extra_joint_selector(
                vertices, self.faces_tensor)
            final_joint_set.append(extra_joints)
        # Create the final joint set
        joints = torch.cat(final_joint_set, dim=1)

        if self.use_joint_regressor:
            reg_joints = torch.einsum(
                'ji,bik->bjk', self.extra_joint_regressor, vertices)
            joints[:, self.source_idxs] = (
                joints[:, self.source_idxs].detach() * 0.0 +
                reg_joints[:, self.target_idxs] * 1.0
            )

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output_joints = KeypointTensor(
            joints, source=self.name, keypoint_names=self.keypoint_names,
            part_indices=self.parts,
            connections=self.connections,
            part_connections=self.part_connections,
        )
        output = defaultdict(lambda: None,
                             joints=output_joints,
                             faces=self.faces)
        if get_skin:
            output['vertices'] = vertices
        if return_full_pose:
            output['full_pose'] = full_pose
        if return_shaped:
            output['v_shaped'] = lbs_output['v_shaped']

        return output


class SMPLH(SMPL):

    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = SMPL.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS
    NAME = 'smplh'

    def __init__(self, model_folder, is_training=False,
                 data_struct=None,
                 gender='male',
                 dtype=torch.float32, single_vertex_per_tip=True,
                 vertex_ids=None, ext='npz',
                 **kwargs):
        """
            Parameters
            ----------
            model_folder: str
                The folder that contains the SMPLH and the MANO
        """
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            model_fn = 'SMPLH_{gender.upper()}.{ext}'
            smplh_path = os.path.join(model_folder, model_fn)
            if ext == 'npz':
                model_data = np.load(smplh_path, allow_pickle=True)
            else:
                with open(smplh_path, 'rb') as smpl_file:
                    model_data = pickle.load(smpl_file, encoding='latin1')
            data_struct = Struct(**model_data)

        super(SMPLH, self).__init__(
            model_folder=model_folder, data_struct=data_struct,
            is_training=is_training,
            vertex_ids=vertex_ids, gender=gender,
            dtype=dtype, single_vertex_per_tip=single_vertex_per_tip,
            **kwargs)

        self.left_hand_pca_mean = data_struct.hands_meanl
        self.right_hand_pca_mean = data_struct.hands_meanr

        self.left_hand_components = data_struct.hands_componentsl
        self.right_hand_components = data_struct.hands_componentsr

    def forward(
        self,
        global_rot: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        betas: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        get_skin: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = False,
        return_shaped: bool = True,
        **kwargs
    ) -> Dict[str, Tensor]:

        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        model_vars = [betas, global_rot, body_pose, transl,
                      left_hand_pose, right_hand_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_rot is None:
            global_rot = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 21, -1, -1).contiguous()
        if left_hand_pose is None:
            left_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if right_hand_pose is None:
            right_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device)

        full_pose = torch.cat([global_rot, body_pose,
                               left_hand_pose,
                               right_hand_pose], dim=1)

        lbs_output = lbs(betas, full_pose, self.v_template,
                         self.shapedirs, self.posedirs,
                         self.J_regressor, self.parents,
                         self.lbs_weights,
                         pose2rot=False, return_shaped=return_shaped)
        vertices = lbs_output['vertices']
        joints = lbs_output['joints']

        final_joint_set = [joints]
        if hasattr(self, 'extra_joint_selector'):
            # Add any extra joints that might be needed
            extra_joints = self.extra_joint_selector(
                vertices, self.faces_tensor)
            final_joint_set.append(extra_joints)
        # Create the final joint set
        joints = torch.cat(final_joint_set, dim=1)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output_joints = KeypointTensor(
            joints, source=self.name, keypoint_names=self.keypoint_names,
            part_indices=self.parts,
            connections=self.connections,
            part_connections=self.part_connections,
        )
        output = defaultdict(lambda: None,
                             joints=output_joints,
                             faces=self.faces)
        if get_skin:
            output['vertices'] = vertices
        if return_full_pose:
            output['full_pose'] = full_pose
        if return_shaped:
            output['v_shaped'] = lbs_output['v_shaped']

        return output


class SMPLX(SMPLH):

    NUM_BODY_JOINTS = SMPLH.NUM_BODY_JOINTS
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    SHAPE_SPACE_DIM = 300
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 12
    HEAD_IDX = 15
    NAME = 'smplx'

    def __init__(self, model_folder, is_training=False,
                 expression=None, use_face_contour=False,
                 gender='neutral',
                 dtype=torch.float32, ext='npz',
                 **kwargs):
        '''
        '''

        model_fn = f'SMPLX_{gender.upper()}.{ext}'
        smplx_path = os.path.join(model_folder, model_fn)
        logger.info(f'Loading model from: {smplx_path}')
        if ext == 'npz':
            data_struct = Struct(**np.load(smplx_path, allow_pickle=True))
        else:
            with open(smplx_path, 'rb') as smplx_file:
                data_struct = Struct(**pickle.load(smplx_file))

        self.use_face_contour = use_face_contour

        super(SMPLX, self).__init__(
            model_folder=model_folder,
            is_training=is_training,
            data_struct=data_struct,
            dtype=dtype,
            gender=gender,
            **kwargs)

        if expression is None:
            expression = {'num': 10}
        self._num_expression_coeffs = expression.get('num', 10)

        # The pickle file that contains the barycentric coordinates for
        # regressing the landmarks
        lmk_faces_idx = data_struct.lmk_faces_idx.astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = data_struct.lmk_bary_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=dtype))

        dynamic_lmk_faces_idx = np.array(
            data_struct.dynamic_lmk_faces_idx, dtype=np.int64)
        dynamic_lmk_faces_idx = torch.tensor(
            dynamic_lmk_faces_idx,
            dtype=torch.long)
        self.register_buffer('dynamic_lmk_faces_idx', dynamic_lmk_faces_idx)

        dynamic_lmk_b_coords = torch.tensor(
            data_struct.dynamic_lmk_bary_coords, dtype=dtype)
        self.register_buffer('dynamic_lmk_bary_coords', dynamic_lmk_b_coords)

        #  neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
        neck_kin_chain = find_joint_kin_chain(self.HEAD_IDX, self.parents)
        self.register_buffer(
            'neck_kin_chain',
            torch.tensor(neck_kin_chain, dtype=torch.long))

        expr_start_idx = self.SHAPE_SPACE_DIM
        expr_end_idx = self.SHAPE_SPACE_DIM + self.num_expression_coeffs
        expr_dirs = data_struct.shapedirs[:, :,
                                          expr_start_idx:expr_end_idx]
        self.register_buffer(
            'expr_dirs', to_tensor(to_np(expr_dirs), dtype=dtype))

    @property
    def num_expression_coeffs(self):
        return self._num_expression_coeffs

    def build_keypoint_names(self) -> StringList:
        model_keypoint_names = super(SMPLX, self).build_keypoint_names()
        # Remove facial contour keypoints if the facial contour is not used
        if not self.use_face_contour:
            model_keypoint_names = [
                n for n in model_keypoint_names if 'contour' not in n]
        return model_keypoint_names

    @property
    def num_body_joints(self):
        return self.NUM_BODY_JOINTS

    @property
    def num_hand_joints(self):
        return self.NUM_HAND_JOINTS

    def extra_repr(self):
        msg = super(SMPLX, self).extra_repr()
        msg = [
            msg,
            f'Number of Expression Coefficients: {self.num_expression_coeffs}',
            f'Use face contour: {self.use_face_contour}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        global_rot: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        betas: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        get_skin: bool = True,
        return_full_pose: bool = False,
        return_shaped: bool = True,
        **kwargs
    ) -> Dict[str, Tensor]:
        ''' SMPL-X forward pass
        '''

        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [betas, global_rot, body_pose, transl,
                      left_hand_pose, right_hand_pose,
                      jaw_pose, leye_pose, reye_pose, expression]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_rot is None:
            global_rot = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 21, -1, -1).contiguous()
        if left_hand_pose is None:
            left_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if right_hand_pose is None:
            right_hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if jaw_pose is None:
            jaw_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if leye_pose is None:
            leye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if reye_pose is None:
            reye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if expression is None:
            expression = torch.zeros([batch_size, self.num_expression_coeffs],
                                     dtype=dtype, device=device)
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)

        full_pose = torch.cat([global_rot, body_pose,
                               jaw_pose, leye_pose, reye_pose,
                               left_hand_pose,
                               right_hand_pose], dim=1)

        # Concatenate the shape and expression coefficients
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        lbs_output = lbs(shape_components, full_pose, self.v_template,
                         shapedirs, self.posedirs,
                         self.J_regressor, self.parents,
                         self.lbs_weights,
                         pose2rot=False,
                         return_shaped=return_shaped)

        vertices = lbs_output['vertices']
        joints = lbs_output['joints']

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size,
                                                                   -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(
            batch_size, -1, -1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_b_coords = (
                find_dynamic_lmk_idx_and_bcoords(
                    vertices, full_pose,
                    self.dynamic_lmk_faces_idx,
                    self.dynamic_lmk_bary_coords,
                    self.neck_kin_chain)
            )

            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_b_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        final_joint_set = [joints, landmarks]
        if hasattr(self, 'extra_joint_selector'):
            # Add any extra joints that might be needed
            extra_joints = self.extra_joint_selector(
                vertices, self.faces_tensor)
            final_joint_set.append(extra_joints)
        # Create the final joint set
        joints = torch.cat(final_joint_set, dim=1)

        if self.use_joint_regressor:
            reg_joints = torch.einsum(
                'ji,bik->bjk', self.extra_joint_regressor, vertices)
            joints[:, self.source_idxs] = (
                joints[:, self.source_idxs].detach() * 0.0 +
                reg_joints[:, self.target_idxs] * 1.0
            )

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output_joints = KeypointTensor(
            joints, source=self.name, keypoint_names=self.keypoint_names,
            part_indices=self.parts,
            connections=self.connections,
            part_connections=self.part_connections,
        )
        output = defaultdict(lambda: None,
                             joints=output_joints,
                             faces=self.faces)
        if get_skin:
            output['vertices'] = vertices
        if return_full_pose:
            output['full_pose'] = full_pose
        if return_shaped:
            output['v_shaped'] = (
                self.v_template + blend_shapes(betas, self.shapedirs))

        return output
