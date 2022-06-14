import os
import sys
from typing import Dict, Tuple, Optional
import os.path as osp

import pickle
import time

from loguru import logger
from collections import defaultdict
from itertools import combinations
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit
from omegaconf import DictConfig

from body_measurements import BodyMeasurements

from .pose_utils import PoseParameterization
from .networks import build_regressor
from .keypoint_loss import KeypointLosses

from ..body_models import KeypointTensor
from ..backbone import build_backbone
from ..camera import CameraParams, build_cam_proj

from human_shape.losses import build_loss
from human_shape.data.structures import StructureList
from human_shape.utils import (
    Tensor, BlendShapeDescription, StringList, AppearanceDescription,
    Timer)

from attributes import A2B, B2A

class HMRLikeRegressor(nn.Module):
    def __init__(
        self,
        body_model_cfg: DictConfig,
        network_cfg: DictConfig,
        loss_cfg: DictConfig,
        dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
        ''' Implements an HMR-like architecture for human body regression
        '''
        super(HMRLikeRegressor, self).__init__()

        # Pose only the final stage of the model to save computation
        self.pose_last_stage = network_cfg.get('pose_last_stage', True)
        logger.info(f'Pose last stage: {self.pose_last_stage}')

        # Build the camera projection function
        camera_cfg = network_cfg.get('camera', {})
        camera_data = build_cam_proj(camera_cfg, dtype=dtype)
        self.projection = camera_data['camera']

        # Retrieve the dimension of the camera and its mean values
        camera_param_dim = camera_data['dim']
        camera_mean = camera_data['mean']
        self.camera_scale_func = camera_data['scale_func']
        camera_space = {'dim': camera_param_dim, 'mean': camera_mean}

        # Create the body model
        self.model = self._build_model(body_model_cfg)

        # Create the decoder, mean vector and dimension for each of the
        # regressed parameters. For most models, these can be split into pose,
        # shape and appearance parameters.
        pose_space = self._build_pose_space(body_model_cfg)
        blendshape_space = self._build_blendshape_space(body_model_cfg)
        appearance_space = self._build_appearance_space(body_model_cfg)

        self.pose_space = pose_space
        self.blendshape_space = blendshape_space
        self.appearance_space = appearance_space

        # Merge all parameter dictionaries
        param_dict = dict(**pose_space, **blendshape_space, **appearance_space)
        param_dict['camera'] = camera_space

        # This list will contain the mean vector for each parameter
        mean_lst = []
        start = 0
        # Iterate over the parameters and their descriptions
        for name, desc in param_dict.items():
            buffer_name = f'{name}_idxs'
            # Compute the indices of the parameters in the predicted flattened
            # array
            indices = list(range(start, start + desc['dim']))
            indices = torch.tensor(indices, dtype=torch.long)
            # Register the indices as a buffer
            self.register_buffer(buffer_name, indices)
            # Append the mean of the current parameter
            mean_lst.append(desc['mean'].view(-1))
            logger.info(f'{name}: {start} -> {start + desc["dim"]}')
            # Update the starting position
            start += desc['dim']
            # Register the parameter mean
            self.register_buffer(f'{name}_mean', desc['mean'])

        # Get the names of the predicted paramters
        self.param_names = list(param_dict.keys())

        param_mean = torch.cat(mean_lst).view(1, -1)
        param_dim = param_mean.numel()
        self._param_dim = param_dim
        self.register_buffer('param_mean', param_mean)

        # Build the feature extraction backbone
        backbone_cfg = network_cfg.get('backbone', {})
        self.backbone, feat_dims = build_backbone(backbone_cfg)

        # Get the key used to select the features to be used
        self.feature_key = network_cfg.get('feature_key', 'avg_pooling')
        feat_dim = feat_dims[self.feature_key]
        self._feat_dim = feat_dim

        # Build the parameter regression network
        self.regressor, num_stages = build_regressor(
            network_cfg, feat_dim, param_dim, param_mean=param_mean)
        logger.info(f'Regressor network: {self.regressor}')
        self._num_stages = num_stages

        # Decide whether
        meas_definition_path = osp.expandvars(network_cfg.get(
            'meas_definition_path', ''))
        meas_vertices_path = osp.expandvars(network_cfg.get(
            'meas_vertices_path', ''))
        compute_measurements = network_cfg.get('compute_measurements', False)
        # Check that the user requested body measurement computation and that
        # the necessary files exist.
        compute_measurements = compute_measurements and osp.exists(
            meas_definition_path) and osp.exists(meas_vertices_path)
        self.compute_measurements = compute_measurements
        logger.info(f'Compute body measurements: {compute_measurements}')
        if self.compute_measurements:
            self.body_measurements = BodyMeasurements(
                {'meas_definition_path': meas_definition_path,
                 'meas_vertices_path': meas_vertices_path},
            )

        # load betas to attributes module
        use_b2a = network_cfg.get('use_b2a', False)
        b2a_males_checkpoint_path = network_cfg.get('b2a_males_checkpoint', '')
        b2a_females_checkpoint_path = network_cfg.get(
            'b2a_females_checkpoint', '')
        self.use_b2a = use_b2a and osp.exists(
            b2a_males_checkpoint_path) and osp.exists(
            b2a_females_checkpoint_path)
        if self.use_b2a:
            logger.info(f'Loading B2A regressors ...')
            male_hparams = torch.load(b2a_males_checkpoint_path)['hyper_parameters']
            male_cfg = male_hparams.get('cfg', {})
            #male_cfg.update(male_hparams)
            # load regressor for males
            self.b2a_males = B2A.load_from_checkpoint(
                b2a_males_checkpoint_path, cfg=male_cfg)
            self.b2a_males.eval()
            for name, params in self.b2a_males.named_parameters():
                params.requires_grad = False
            # load regressor for males
            female_hparams = torch.load(b2a_females_checkpoint_path)['hyper_parameters']
            female_cfg = female_hparams.get('cfg', {})
            #female_cfg.update(female_hparams)
            self.b2a_females = B2A.load_from_checkpoint(
                b2a_females_checkpoint_path, cfg=female_cfg)
            self.b2a_females.eval()
            for name, params in self.b2a_females.named_parameters():
                params.requires_grad = False

        # load a2b regressor
        use_a2b = network_cfg.get('use_a2b', False)
        self.num_attributes = network_cfg.get('num_attributes', False)
        a2b_males_checkpoint_path = network_cfg.get('a2b_males_checkpoint', '')
        a2b_females_checkpoint_path = network_cfg.get(
            'a2b_females_checkpoint', '')
        self.use_a2b = use_a2b and osp.exists(
            a2b_males_checkpoint_path) and osp.exists(
            a2b_females_checkpoint_path)
        if self.use_a2b:
            logger.info(f'Loading A2B regressors ...')
            # load regressor for males
            self.a2b_males = A2B.load_from_checkpoint(
                a2b_males_checkpoint_path)
            self.a2b_males.renderer = None
            if hasattr(self.a2b_males, 'hd_operator'):
                delattr(self.a2b_males, 'hd_operator')
            self.a2b_males.eval()

            for name, params in self.a2b_males.named_parameters():
                params.requires_grad = False

            # load regressor for males
            self.a2b_females = A2B.load_from_checkpoint(
                a2b_females_checkpoint_path)
            if hasattr(self.a2b_females, 'hd_operator'):
                delattr(self.a2b_females, 'hd_operator')
            self.a2b_females.renderer = None
            self.a2b_females.eval()
            for name, params in self.a2b_females.named_parameters():
                params.requires_grad = False

        # Build the losses
        self._build_losses(loss_cfg)

        #  self.timer = Timer(name='Measurements', sync=True, verbose=True)

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def get_mean(self) -> Tensor:
        return self.param_mean

    @property
    def feat_dim(self) -> int:
        ''' Returns the dimension of the expected feature vector '''
        return self._feat_dim

    @property
    def num_stages(self) -> int:
        ''' Returns the number of stages for the iterative predictor'''
        return self._num_stages

    @property
    def num_betas(self) -> int:
        ''' Returns the number of stages for the iterative predictor'''
        return self.model.num_betas

    @property
    def num_expression_coeffs(self) -> int:
        ''' Returns the number of stages for the iterative predictor'''
        if hasattr(self.model, 'num_expression_coeffs'):
            return self.model.num_expression_coeffs
        else:
            return 0

    def flat_params_to_dict(self, param_tensor: Tensor) -> Dict[str, Tensor]:
        ''' Convert a flat parameter tensor to a dictionary of parameters
        '''
        param_dict = {}
        for name in self.param_names:
            indices = getattr(self, f'{name}_idxs')
            param_dict[name] = torch.index_select(param_tensor, 1, indices)
            logger.debug(f'{name}: {param_dict[name].shape}')
        return param_dict

    def _build_losses(self, loss_cfg):
        # Build the keypoint losses
        self.keypoint_loss_modules = nn.ModuleDict()
        keypoint_loss_modules, center_around = self._build_keypoint_losses(
            loss_cfg)
        self.center_around = center_around
        for name, module in keypoint_loss_modules.items():
            self.keypoint_loss_modules[name] = module
            logger.info(f'{name}: {module}')

        # Build parameter losses and regularizers
        self.param_loss, self.regularizer = self._build_parameter_losses(
            loss_cfg)
        logger.info(f'Parameter loss: {self.param_loss}')
        logger.info(f'Regularizer: {self.regularizer}')

        # Computes the loss between the ground-truth and the estimated mesh
        # edges.
        self.loss_activ_step = {}
        edge_loss_cfg = loss_cfg.get('edge', {})
        self.mesh_edge_weight = edge_loss_cfg.get('weight', 0.0)
        if self.mesh_edge_weight > 0:
            self.mesh_edge_loss = build_loss(**edge_loss_cfg)
            self.loss_activ_step['edge'] = edge_loss_cfg.get('enable', 0)
            logger.debug('3D Mesh Edges weight, loss: {}, {}',
                         self.mesh_edge_weight, self.mesh_edge_loss)

        # Computes the loss between the ground-truth and the estimated mesh
        # vertices.
        vertex_loss_cfg = loss_cfg.get('vertex', {})
        self.mesh_vertex_weight = vertex_loss_cfg.get('weight', 0.0)
        if self.mesh_vertex_weight > 0:
            self.mesh_vertex_loss = build_loss(**vertex_loss_cfg)
            self.loss_activ_step['vertex'] = vertex_loss_cfg.get('enable', 0)
            logger.debug('3D Mesh vertices weight, loss: {}, {}',
                         self.mesh_vertex_weight, self.mesh_vertex_loss)

        # Loss between the mass computed from the mesh and the ground-truth
        mass_loss_cfg = loss_cfg.get('mass', {})
        self.mass_weight = mass_loss_cfg.get('weight', 0.0)
        if self.mass_weight > 0:
            self.mass_loss = build_loss(**mass_loss_cfg)
            self.loss_activ_step['mass'] = mass_loss_cfg.get('enable', 0)
            logger.debug('Mass weight, loss: {}, {}',
                         self.mass_weight, self.mass_loss)

        height_loss_cfg = loss_cfg.get('height', {})
        self.height_weight = height_loss_cfg.get('weight', 0.0)
        if self.height_weight > 0:
            self.height_loss = build_loss(**height_loss_cfg)
            self.loss_activ_step['height'] = height_loss_cfg.get('enable', 0)
            logger.debug('Height weight, loss: {}, {}',
                         self.height_weight, self.height_loss)

        chest_loss_cfg = loss_cfg.get('chest', {})
        self.chest_weight = chest_loss_cfg.get('weight', 0.0)
        if self.chest_weight > 0:
            self.chest_loss = build_loss(**chest_loss_cfg)
            self.loss_activ_step['chest'] = chest_loss_cfg.get('enable', 0)
            logger.debug('Chest weight, loss: {}, {}',
                         self.chest_weight, self.chest_loss)

        waist_loss_cfg = loss_cfg.get('waist', {})
        self.waist_weight = waist_loss_cfg.get('weight', 0.0)
        if self.waist_weight > 0:
            self.waist_loss = build_loss(**waist_loss_cfg)
            self.loss_activ_step['waist'] = waist_loss_cfg.get('enable', 0)
            logger.debug('Waist weight, loss: {}, {}',
                         self.waist_weight, self.waist_loss)

        hips_loss_cfg = loss_cfg.get('hips', {})
        self.hips_weight = hips_loss_cfg.get('weight', 0.0)
        if self.hips_weight > 0:
            self.hips_loss = build_loss(**hips_loss_cfg)
            self.loss_activ_step['hips'] = hips_loss_cfg.get('enable', 0)
            logger.debug('Hips weight, loss: {}, {}',
                         self.hips_weight, self.hips_loss)

        # Identity loss
        identity_loss_cfg = loss_cfg.get('identity', {})
        self.identity_weight = identity_loss_cfg.get('weight', 0.0)
        if self.identity_weight > 0:
            self.identity_loss = build_loss(**identity_loss_cfg)
            self.loss_activ_step['identity'] = identity_loss_cfg.get(
                'enable', 0)
            logger.debug('Identity weight, loss: {}, {}',
                         self.identity_weight, self.identity_loss)

        attribute_loss_cfg = loss_cfg.get('attributes')
        self.attribute_weight = attribute_loss_cfg.get('weight', 0.0)
        if self.attribute_weight > 0:
            self.attribute_loss = nn.MSELoss(reduction='mean')
            # build_loss(**attribute_loss_cfg)
            self.loss_activ_step['attributes'] = attribute_loss_cfg.get(
                'enable', 0)
            logger.debug('Attribute weight, loss {}, {}',
                         self.attribute_weight, self.attribute_loss)

        # loss on regressor betas and refind betas
        beta_ref_loss_cfg = loss_cfg.get('beta_refined')
        self.beta_ref_weight = beta_ref_loss_cfg.get('weight', 0.0)
        if self.beta_ref_weight > 0:
            # loss for betas
            self.beta_ref_loss = nn.MSELoss(reduction='mean')
            # build_loss(**attribute_loss_cfg)
            self.loss_activ_step['beta_refined'] = beta_ref_loss_cfg.get(
                'enable', 0)
            logger.debug('Beta refined weight, loss {}, {}',
                         self.beta_ref_weight, self.beta_ref_loss)

        # loss on regressor vertices and refined vertices
        vertex_ref_loss_cfg = loss_cfg.get('vertex_refined')
        self.mesh_vertex_ref_weight = vertex_ref_loss_cfg.get('weight', 0.0)
        if self.mesh_vertex_ref_weight > 0:
            self.mesh_vertex_ref_loss = build_loss(**vertex_ref_loss_cfg)
            self.loss_activ_step['vertex_ref'] = vertex_ref_loss_cfg.get(
                'enable', 0)
            logger.debug('3D Mesh vertices weight, loss: {}, {}',
                         self.mesh_vertex_ref_weight, self.mesh_vertex_ref_loss)

    def compute_losses(
        self,
        targets: StructureList,
        out_params: Dict[str, Tensor],
        proj_joints: Optional[KeypointTensor] = None,
        est_joints3d: Optional[KeypointTensor] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        ''' Computes the losses
        '''

        num_stages = out_params.get('num_stages', 1)
        stage_keys = out_params.get('stage_keys', [])

        last_stage = out_params[stage_keys[-1]]
        if proj_joints is None:
            proj_joints = out_params.get('proj_joints')
        if est_joints3d is None:
            est_joints3d = last_stage.get('joints')

        losses = {}
        # Map all ground-truth joints to the current model format
        gt_joints2d = torch.stack(
            [t.to_dset(
                target_dataset=est_joints3d.source,
                target_names=est_joints3d.keypoint_names)
             for t in targets])

        # Store the indices of the batch elements that have ground-truth 3D
        # keypoint annotations.
        gt_joints_3d_indices = torch.tensor(
            [ii for ii, t in enumerate(targets)
             if t.has_field('keypoints3d')],
            dtype=torch.long, device=device)
        gt_joints3d = None
        if len(gt_joints_3d_indices) > 0:
            # Stack all 3D joint tensors
            gt_joints3d = torch.stack(
                [t.get_field('keypoints3d').to_dset(
                    target_dataset=est_joints3d.source,
                    target_names=est_joints3d.keypoint_names,
                ) for t in targets if t.has_field('keypoints3d')])

        keypoint_losses = {}
        # Compute the keypoint losses
        for part_key, loss in self.keypoint_loss_modules.items():
            part_name = part_key.replace('_keypoint_loss', '')
            keypoint_losses[part_key] = loss(
                proj_joints,
                gt_joints2d,
                est_joints3d[gt_joints_3d_indices],
                gt_joints3d,
                part=part_name,
                center_around=self.center_around[part_name],
            )
            for key, value in keypoint_losses[part_key].items():
                losses[f'{part_name}_{key}'] = value
        #  losses.update(keypoint_losses)

        # Compute the vertex loss
        est_vertices = out_params[f'stage_{num_stages - 1:02d}'][
            'vertices']
        has_vertices = torch.tensor(
            [t.has_field('vertices') for t in targets], device=device,
            dtype=torch.bool)
        # If there are ground-truth vertices
        if has_vertices.sum().item() > 0:
            # Get all ground truth vertices
            gt_vertices = torch.stack([
                t.get_field('vertices').vertices for t in targets
                if t.has_field('vertices')])
            # Compute the edge loss
            if self.mesh_edge_weight > 0:
                edge_loss_val = self.mesh_edge_loss(
                    gt_vertices=gt_vertices,
                    est_vertices=est_vertices[has_vertices])
                losses['mesh_edge_loss'] = (
                    self.mesh_edge_weight * edge_loss_val)
            # Compute the vertex loss
            if self.mesh_vertex_weight > 0:
                vertex_loss_val = self.mesh_vertex_loss(
                    est_vertices[has_vertices], gt_vertices)
                losses['mesh_vertex_loss'] = (
                    self.mesh_vertex_weight * vertex_loss_val)

        # Compute the parameter losses
        parameter_losses = self.param_loss(
            out_params,
            targets,
            gt_confs=gt_joints2d[:, :, -1],
            keypoint_part_indices=est_joints3d._part_indices,
            device=device,
        )
        for key, value in parameter_losses.items():
            losses[key] = value

        reg_losses = self.regularizer(out_params, targets=targets)
        for key, value in reg_losses.items():
            losses[key] = value

        if self.compute_measurements:
            # Compute measurement losses
            for meas_name, meas_data in out_params['measurements'].items():
                loss_weight = getattr(self, f'{meas_name}_weight')
                if loss_weight == 0:
                    continue
                loss = getattr(self, f'{meas_name}_loss')
                indices = []
                gt_values = []
                for ii, t in enumerate(targets):
                    if t.has_field(meas_name):
                        meas_value = t.get_field(meas_name)
                        # Make sure that the measurement is within a valid
                        # range
                        if meas_value > 0:
                            indices.append(ii)
                            gt_values.append(meas_value)

                if len(gt_values) > 0 and loss_weight > 0:
                    dtype = est_vertices.dtype
                    gt_values = torch.tensor(
                        gt_values, device=device, dtype=dtype)
                    est_values = meas_data
                    if not torch.is_tensor(est_values):
                        est_values = torch.stack(est_values)
                    losses[meas_name] = loss_weight * loss(
                        est_values[indices], gt_values)

        if self.identity_weight > 0:
            indices, identities = [], []
            for ii, t in enumerate(targets):
                if t.has_field('identity'):
                    indices.append(ii)
                    identities.append(t.get_field('identity'))

            indices = np.array(indices)
            identities = np.array(identities)
            unique = np.unique(identities)

            num_stages = out_params['num_stages']
            last_stage_key = f'stage_{num_stages - 1:02d}'
            last_params = out_params[last_stage_key]

            betas = last_params['betas']

            losses['identity'] = 0.0
            # Iterate over identities
            for identity in unique:
                mask = (identities == identity)
                if mask.sum() <= 1:
                    continue
                # Get the original indices
                ii = torch.from_numpy(indices[np.where(mask)[0]]).to(
                    device=device)

                # Compute all identity pairs
                pairs = torch.tensor(list(combinations(ii, 2)))

                # Compute the loss between the pairs
                losses['identity'] += self.identity_loss(
                    betas[pairs[0]], pairs[[1]])

        if self.attribute_weight > 0:
            indices, gt_attributes = [], []
            for ii, t in enumerate(targets):
                if t.has_field('attributes'):
                    indices.append(ii)
                    gt_attributes.append(t.get_field('attributes'))
            pred_attributes = out_params['attributes']
            gt_attributes = torch.tensor(
                gt_attributes, device=device)
            losses['attribute'] = self.attribute_weight * self.attribute_loss(
                pred_attributes[indices], gt_attributes)
            # check larger zero

        if self.use_a2b:
            indices = []
            for ii, t in enumerate(targets):
                if t.has_field('attributes'):
                    indices.append(ii)

            # get the last stage params
            num_stages = out_params['num_stages']
            last_stage_key = f'stage_{num_stages -1:02d}'
            last_params = out_params[last_stage_key]

            # betas loss
            ref_betas = last_params['betas_ref']
            betas = last_params['betas']
            if self.beta_ref_weight > 0:
                losses['beta_ref'] = self.beta_ref_weight * self.beta_ref_loss(
                    ref_betas[indices], betas[indices])

            # v2v loss
            v_shaped_ref = last_params['v_shaped_ref']
            v_shaped = last_params['v_shaped']
            if self.mesh_vertex_ref_weight > 0:
                value = self.mesh_vertex_ref_loss(
                    v_shaped[indices], v_shaped_ref[indices])
                val = (v_shaped[indices] - v_shaped_ref[indices]).pow(2).view(
                    len(indices), -1).sum(dim=-1).mean()
                losses['vertex_ref'] = (
                    self.mesh_vertex_ref_weight * self.mesh_vertex_ref_loss(
                        v_shaped[indices], v_shaped_ref[indices])
                )

        # lossl = [f'{v.item():.2f}' for k,v in losses.items()]
        # print(lossl)

        return losses

    def _build_keypoint_losses(
        self,
        loss_cfg
    ) -> Tuple[Dict[str, KeypointLosses], Dict[str, StringList]]:
        return {}, {}

    def _build_pose_space(
        self, body_model_cfg
    ) -> Dict[str, PoseParameterization]:
        mean_pose_path = osp.expandvars(self.curr_model_cfg.mean_pose_path)
        self.mean_poses_dict = {}
        if osp.exists(mean_pose_path):
            logger.debug('Loading mean pose from: {} ', mean_pose_path)
            with open(mean_pose_path, 'rb') as f:
                self.mean_poses_dict = pickle.load(f)
        return {}

    def _build_appearance_space(
        self, body_model_cfg,
        dtype=torch.float32,
    ) -> Dict[str, AppearanceDescription]:
        return {}

    def _build_blendshape_space(
        self, body_model_cfg,
        dtype=torch.float32,
    ) -> Dict[str, BlendShapeDescription]:
        return {}

    def compute_features(
        self,
        images: Tensor,
        extra_features: Optional[Tensor] = None,
    ) -> Tensor:
        ''' Computes features for the current input
        '''
        feat_dict = self.backbone(images)
        features = feat_dict[self.feature_key]
        return features

    def forward(
        self,
        images: Tensor,
        targets: StructureList = None,
        compute_losses: bool = True,
        cond: Optional[Tensor] = None,
        extra_features: Optional[Tensor] = None,
        **kwargs
    ):
        batch_size = len(images)
        device, dtype = images.device, images.dtype

        # Compute the features
        features = self.compute_features(images, extra_features=extra_features)

        regr_output = self.regressor(
            features, cond=cond, extra_features=extra_features)

        if torch.is_tensor(regr_output):
            parameters = [regr_output]
        elif isinstance(regr_output, (tuple, list)):
            parameters = regr_output[0]

        param_dicts = []
        # Iterate over the estimated parameters and decode them. For example,
        # rotation predictions need to be converted from whatever format is
        # predicted by the network to rotation matrics.
        for ii, params in enumerate(parameters):
            curr_params_dict = self.flat_params_to_dict(params)
            out_dict = {}
            for key, val in curr_params_dict.items():
                if hasattr(self, f'{key}_decoder'):
                    decoder = getattr(self, f'{key}_decoder')
                    out_dict[key] = decoder(val)
                    out_dict[f'raw_{key}'] = val.clone()
                else:
                    out_dict[key] = val
            param_dicts.append(out_dict)

        num_stages = len(param_dicts)

        if self.pose_last_stage:
            merged_params = param_dicts[-1]
        else:
            # If we want to pose all prediction stages to visualize the meshes,
            # then it is much faster to concatenate all parameters, pose and
            # split, instead of running the skinning function N times.
            merged_params = {}
            for key in param_dicts[0].keys():
                param = []
                for ii in range(num_stages):
                    if param_dicts[ii][key] is None:
                        continue
                    param.append(param_dicts[ii][key])
                merged_params[key] = torch.cat(param, dim=0)

        # Compute the body surface using the current estimation of the pose and
        # the shape
        model_output = self.model(
            get_skin=True, return_shaped=True, **merged_params)

        # Split the vertices, joints, etc. to stages
        out_params = defaultdict(lambda: dict())
        for key in model_output:
            if isinstance(model_output[key], (KeypointTensor,)):
                curr_val = model_output[key]
                out_list = torch.split(curr_val._t, batch_size, dim=0)
                if len(out_list) == num_stages:
                    for ii, value in enumerate(out_list):
                        out_params[f'stage_{ii:02d}'][key] = (
                            KeypointTensor.from_obj(value, curr_val)
                        )
                else:
                    # Else add only the last
                    out_key = f'stage_{num_stages - 1:02d}'
                    out_params[out_key][key] = KeypointTensor.from_obj(
                        out_list[-1], curr_val)
            elif torch.is_tensor(model_output[key]):
                curr_val = model_output[key]
                out_list = torch.split(curr_val, batch_size, dim=0)
                # If the number of outputs is equal to the number of stages
                # then store each stage
                if len(out_list) == num_stages:
                    for ii, value in enumerate(out_list):
                        out_params[f'stage_{ii:02d}'][key] = value
                else:
                    # Else add only the last
                    out_key = f'stage_{num_stages - 1:02d}'
                    out_params[out_key][key] = out_list[-1]

        # Extract the estimated camera parameters
        camera_params = param_dicts[-1]['camera']
        scale = camera_params[:, 0].view(-1, 1)
        translation = camera_params[:, 1:3]
        # Pass the predicted scale through exp() to make sure that the
        # scale values are always positive
        scale = self.camera_scale_func(scale)

        est_joints3d = out_params[f'stage_{num_stages - 1:02d}']['joints']
        # Project the joints on the image plane
        proj_joints = self.projection(
            est_joints3d, scale=scale, translation=translation)

        # Add the projected joints
        out_params['proj_joints'] = proj_joints
        out_params['num_stages'] = num_stages
        out_params['features'] = features

        out_params['camera_parameters'] = CameraParams(
            translation=translation, scale=scale,
            scale_first=getattr(self.projection, 'scale_first', False))

        stage_keys = []
        for n in range(num_stages):
            stage_key = f'stage_{n:02d}'
            stage_keys.append(stage_key)
            out_params[stage_key]['faces'] = model_output['faces']
            out_params[stage_key].update(param_dicts[n])

        if self.compute_measurements:
            last_stage_key = f'stage_{num_stages - 1:02d}'
            shaped_triangles = out_params[last_stage_key]['v_shaped'][
                :, self.model.faces_tensor]

            # Compute the measurements on the body
            measurements = self.body_measurements(
                shaped_triangles)['measurements']

            meas_dict = {}
            for name, d in measurements.items():
                meas_dict[name] = d['tensor']

            out_params[last_stage_key].update(measurements=meas_dict)
            out_params.update(measurements=meas_dict)

        out_params['stage_keys'] = stage_keys
        out_params[stage_keys[-1]]['proj_joints'] = proj_joints

        if self.use_b2a:
            genders = [x.get_field('gender') if x.has_field(
                'gender') else None for x in targets]
            genders = np.array(
                [x.lower()[0] if (x is not None and x != '') else 'n' for x in genders])
            genders_males = np.where(genders == 'm')[0]
            genders_females = np.where(genders == 'f')[0]
            betas = parameters[-1][:, self.betas_idxs]
            # attributes = self.b2a(betas)
            attributes_males = self.b2a_males(betas[genders_males, :])
            attributes_females = self.b2a_females(betas[genders_females, :])
            attributes = torch.zeros(
                betas.shape[0], attributes_males.shape[1]).to(betas.device)
            attributes[genders_males, :] = attributes_males
            attributes[genders_females, :] = attributes_females
            out_params['attributes'] = attributes

        if self.use_a2b:
            last_stage_key = f'stage_{num_stages - 1:02d}'
            genders = [x.get_field('gender') if x.has_field('gender')
                       else None for x in targets]
            genders = np.array([x.lower()[0] if (x is not None and x != '')
                                else 'n' for x in genders])
            genders_males = np.where(genders == 'm')[0]
            genders_females = np.where(genders == 'f')[0]
            attr = torch.stack([torch.tensor(x.get_field('attributes'))
                                if x.has_field('attributes')
                                else torch.zeros(self.num_attributes)
                                for x in targets]).to(device)

            # Get the height and weight values from the targets and convert them
            # to a single tensor. If a value is not defined, then simply use a
            # "mean" value from the link below
            # Source: https://ourworldindata.org/human-height#how-does-human-height-vary-across-the-world

            male_height = torch.tensor([
                t.get_field('height', 1.71)
                for t in targets], dtype=torch.float32, device=device)
            female_height = torch.tensor([
                t.get_field('height', 1.59)
                for t in targets], dtype=torch.float32, device=device)

            male_weight = torch.tensor([
                t.get_field('weight', 71.0)
                for t in targets], dtype=torch.float32, device=device)
            female_weight = torch.tensor([
                t.get_field('weight', 62.0)
                for t in targets], dtype=torch.float32, device=device)

            female_input_vec = {
                'rating': attr,
                'height_gt': female_height,
                'weight_gt': female_weight,
                'height_bg': measurements['height']['tensor'],
                'weight_bg': measurements['mass']['tensor'],
            }
            # Convert the female attributes, height and weight to a vector
            # to a feature vector
            female_feature_vec, _ = self.a2b_females.create_input_feature_vec(
                female_input_vec)

            male_input_vec = {
                'rating': attr,
                'height_gt': male_height,
                'weight_gt': male_weight,
                'height_bg': measurements['height']['tensor'],
                'weight_bg': measurements['mass']['tensor'],
            }
            # Convert the male attributes, height and weight to a vector
            # to a feature vector
            male_feature_vec, _ = self.a2b_males.create_input_feature_vec(
                male_input_vec)

            # get the refined betas from A2S
            betas_ref = torch.zeros((
                batch_size, self.betas_idxs.shape[0]), device=device)

            betas_ref_males = self.a2b_males(
                male_feature_vec[genders_males, :])

            # logger.info(self.a2b_males)
            betas_ref_females = self.a2b_females(
                female_feature_vec[genders_females, :])
            betas_ref[genders_males, :] = betas_ref_males
            betas_ref[genders_females, :] = betas_ref_females
            merged_params['betas'] = betas_ref

            # Compute only the blend-shapes, much faster than the full forward
            # pass
            v_shaped_ref = self.model.forward_shape(betas_ref)['v_shaped']
            out_params[last_stage_key]['betas_ref'] = betas_ref
            out_params[last_stage_key]['v_shaped_ref'] = v_shaped_ref

        losses = {}
        if self.training and compute_losses:
            losses = self.compute_losses(
                targets, proj_joints=proj_joints,
                est_joints3d=est_joints3d,
                out_params=out_params, device=device,
            )
        out_params['losses'] = losses
        #  for key, val in out_params.items():
        #  if torch.is_tensor(val):
        #  logger.info(f'{key}: {val.shape}')
        #  elif isinstance(val, (dict,)):
        #  for k, v in val.items():
        #  if torch.is_tensor(v):
        #  logger.info(f'{key}, {k}: {v.shape}')

        return out_params
