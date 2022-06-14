import sys
import os.path as osp
from typing import Dict, Tuple, Optional
from loguru import logger

import math
import numpy as np
import torch
import torch.nn as nn

from omegaconf import DictConfig

from ..common.iterative_regressor import HMRLikeRegressor
from ..common.keypoint_loss import KeypointLosses
from ..common.pose_utils import (
    build_pose_parameterization, PoseParameterization)

from ..body_models import build_body_model, KeypointTensor

from .body_loss_modules import (
    SMPLLossModule, SMPLHLossModule, SMPLXLossModule,
    RegularizerModule,
)
from .registry import BODY_HEAD_REGISTRY

from human_shape.data.structures import StructureList
from human_shape.utils import Tensor, BlendShapeDescription, StringList

__all__ = [
    'SMPLRegressor',
    'SMPLHRegressor',
    'SMPLXRegressor',
]


@BODY_HEAD_REGISTRY.register()
class SMPLRegressor(HMRLikeRegressor):

    def __init__(
        self, body_model_cfg, network_cfg, loss_cfg,
        dtype=torch.float32,
    ) -> None:
        super(SMPLRegressor, self).__init__(
            body_model_cfg, network_cfg, loss_cfg)

    def _build_model(self, body_model_cfg):
        # The configuration for all body models
        self.body_model_cfg = body_model_cfg
        model = build_body_model(body_model_cfg)
        self.model_type = model.name

        # The config of the model
        self.curr_model_cfg = body_model_cfg.get(self.model_type, {})

        logger.info(f'Body model: {model}')
        return model

    def _build_keypoint_losses(
        self, loss_cfg
    ) -> Tuple[Dict[str, KeypointLosses], Dict[str, StringList]]:
        loss_modules, center_around = super(
            SMPLRegressor, self)._build_keypoint_losses(loss_cfg)
        KEYPOINT_PARTS = ['body']

        center_around['body'] = ['left_hip', 'right_hip']
        for part in KEYPOINT_PARTS:
            keyp_loss = KeypointLosses(
                loss_cfg.get(f'{part}_joints_2d'),
                loss_cfg.get(f'{part}_joints_3d'),
                loss_cfg.get(f'{part}_edge_2d', {}),
                loss_cfg.get(f'{part}_edge_3d', {})
            )
            loss_modules[f'{part}_keypoint_loss'] = keyp_loss
        return loss_modules, center_around

    def _build_parameter_losses(
        self, loss_cfg
    ) -> [SMPLLossModule, RegularizerModule]:
        return SMPLLossModule(loss_cfg), RegularizerModule(
            loss_cfg,
            body_pose_mean=self.mean_poses_dict.get('body_pose'),
        )

    def _build_pose_space(
        self,
        body_model_cfg: DictConfig,
    ) -> Dict[str, PoseParameterization]:
        param_desc = super(SMPLRegressor, self)._build_pose_space(
            body_model_cfg)

        global_rot_desc = build_pose_parameterization(
            1, **self.curr_model_cfg.global_rot)
        self.global_rot_decoder = global_rot_desc.decoder

        body_pose_desc = build_pose_parameterization(
            num_angles=self.model.num_body_joints,
            mean=self.mean_poses_dict.get('body_pose', None),
            **self.curr_model_cfg.body_pose)
        self.body_pose_decoder = body_pose_desc.decoder

        global_rot_type = body_model_cfg.get('global_rot', {}).get(
            'param_type', 'cont_rot_repr')
        # Rotate the model 180 degrees around the x-axis
        global_rot_mean = global_rot_desc.mean
        if global_rot_type == 'aa':
            global_rot_mean[0] = math.pi
        elif global_rot_type == 'cont_rot_repr':
            global_rot_mean[3] = -1
        return {
            'global_rot': global_rot_desc,
            'body_pose': body_pose_desc,
        }

    def _build_blendshape_space(
            self, body_model_cfg, dtype=torch.float32
    ) -> Dict[str, BlendShapeDescription]:
        blendshape_desc = super(SMPLRegressor, self)._build_blendshape_space(
            body_model_cfg, dtype=dtype)
        num_betas = self.model.num_betas

        shape_mean_path = body_model_cfg.get('shape_mean_path', '')
        shape_mean_path = osp.expandvars(shape_mean_path)
        if osp.exists(shape_mean_path):
            shape_mean = torch.from_numpy(
                np.load(shape_mean_path, allow_pickle=True)).to(
                dtype=dtype).reshape(1, -1)[:, :num_betas].reshape(-1)
        else:
            shape_mean = torch.zeros([num_betas], dtype=dtype)
        shape_desc = BlendShapeDescription(dim=num_betas, mean=shape_mean)
        blendshape_desc['betas'] = shape_desc
        return blendshape_desc


@BODY_HEAD_REGISTRY.register()
class SMPLHRegressor(SMPLRegressor):
    def __init__(
        self,
        body_model_cfg: DictConfig,
        network_cfg: DictConfig,
        loss_cfg: DictConfig,
        dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
        ''' SMPL+H Regressor
        '''
        self.predict_hands = network_cfg.get('predict_hands', True)
        logger.info(f'Predict hands: {self.predict_hands}')
        super(SMPLHRegressor, self).__init__(
            body_model_cfg, network_cfg, loss_cfg, dtype=dtype)

    def _build_pose_space(
        self,
        body_model_cfg
    ) -> Dict[str, PoseParameterization]:
        param_desc = super(SMPLHRegressor, self)._build_pose_space(
            body_model_cfg)
        if self.predict_hands:
            left_hand_cfg = self.curr_model_cfg.get('left_hand_pose', {})
            right_hand_cfg = self.curr_model_cfg.get('right_hand_pose', {})
            left_hand_pose_desc = build_pose_parameterization(
                num_angles=self.model.num_hand_joints,
                pca_basis=self.model.left_hand_components,
                mean=self.mean_poses_dict.get('left_hand_pose', None),
                **left_hand_cfg)
            logger.debug(
                'Left hand pose decoder: {}', left_hand_pose_desc.decoder)
            param_desc['left_hand_pose'] = left_hand_pose_desc
            self.left_hand_pose_decoder = left_hand_pose_desc.decoder

            right_hand_pose_desc = build_pose_parameterization(
                num_angles=15,
                mean=self.mean_poses_dict.get('right_hand_pose', None),
                pca_basis=self.model.right_hand_components,
                **right_hand_cfg)
            self.right_hand_pose_decoder = right_hand_pose_desc.decoder
            logger.debug(
                'Right hand pose decoder: {}', right_hand_pose_desc.decoder)
            param_desc['right_hand_pose'] = right_hand_pose_desc
        return param_desc

    def _build_parameter_losses(
        self, loss_cfg
    ) -> [SMPLHLossModule, RegularizerModule]:
        return SMPLHLossModule(loss_cfg), RegularizerModule(
            loss_cfg,
            body_pose_mean=self.mean_poses_dict.get('body_pose'),
            left_hand_pose_mean=self.mean_poses_dict.get('left_hand_pose'),
            right_hand_pose_mean=self.mean_poses_dict.get('right_hand_pose'),
        )

    def _build_keypoint_losses(
        self, loss_cfg
    ) -> Tuple[Dict[str, KeypointLosses], Dict[str, StringList]]:
        loss_modules, center_around = super(
            SMPLHRegressor, self)._build_keypoint_losses(loss_cfg)

        KEYPOINT_PARTS = ['left_hand', 'right_hand']
        center_around['left_hand'] = ['left_wrist']
        center_around['right_hand'] = ['right_wrist']
        for part in KEYPOINT_PARTS:
            keyp_loss = KeypointLosses(
                loss_cfg.get(f'{part}_joints_2d'),
                loss_cfg.get(f'{part}_joints_3d'),
                loss_cfg.get(f'{part}_edge_2d', {}),
                loss_cfg.get(f'{part}_edge_3d', {})
            )
            loss_modules[f'{part}_keypoint_loss'] = keyp_loss

        return loss_modules, center_around


@BODY_HEAD_REGISTRY.register()
class SMPLXRegressor(SMPLHRegressor):
    def __init__(
        self,
        body_model_cfg: DictConfig,
        network_cfg: DictConfig,
        loss_cfg: DictConfig,
        dtype: Optional[torch.dtype] = torch.float32
    ):
        self.predict_face = network_cfg.get('predict_face', True)
        logger.info(f'Predict face: {self.predict_face}')
        super(SMPLXRegressor, self).__init__(
            body_model_cfg, network_cfg, loss_cfg, dtype=dtype)

    def _build_pose_space(
        self, body_model_cfg
    ) -> Dict[str, PoseParameterization]:
        param_desc = super(SMPLXRegressor, self)._build_pose_space(
            body_model_cfg)
        if self.predict_face:
            jaw_pose_desc = build_pose_parameterization(
                1, mean=self.mean_poses_dict.get('jaw_pose', None),
                **self.curr_model_cfg.jaw_pose)
            self.jaw_pose_decoder = jaw_pose_desc.decoder
            logger.debug('Jaw pose decoder: {}', jaw_pose_desc.decoder)
            param_desc['jaw_pose'] = jaw_pose_desc
        return param_desc

    def _build_blendshape_space(
            self, body_model_cfg,
            dtype=torch.float32
    ) -> Dict[str, BlendShapeDescription]:
        blendshape_desc = super(SMPLXRegressor, self)._build_blendshape_space(
            body_model_cfg, dtype=dtype)
        if self.predict_face:
            # The number of expression coefficients
            num_expression_coeffs = self.model.num_expression_coeffs
            self._num_expression_coeffs = num_expression_coeffs
            expression_mean = torch.zeros([num_expression_coeffs], dtype=dtype)

            blendshape_desc['expression'] = BlendShapeDescription(
                dim=num_expression_coeffs, mean=expression_mean)
        return blendshape_desc

    def _build_keypoint_losses(
        self, loss_cfg
    ) -> Tuple[Dict[str, KeypointLosses], Dict[str, StringList]]:
        loss_modules, center_around = super(
            SMPLXRegressor, self)._build_keypoint_losses(loss_cfg)

        KEYPOINT_PARTS = ['face']
        center_around['face'] = ['left_hip', 'right_hip']
        for part in KEYPOINT_PARTS:
            keyp_loss = KeypointLosses(
                loss_cfg.get(f'{part}_joints_2d'),
                loss_cfg.get(f'{part}_joints_3d'),
                loss_cfg.get(f'{part}_edge_2d', {}),
                loss_cfg.get(f'{part}_edge_3d', {})
            )
            loss_modules[f'{part}_keypoint_loss'] = keyp_loss

        return loss_modules, center_around

    def _build_parameter_losses(
        self, loss_cfg
    ) -> [SMPLXLossModule, RegularizerModule]:
        return SMPLXLossModule(loss_cfg), RegularizerModule(
            loss_cfg,
            body_pose_mean=self.mean_poses_dict.get('body_pose'),
            left_hand_pose_mean=self.mean_poses_dict.get('left_hand_pose'),
            right_hand_pose_mean=self.mean_poses_dict.get('right_hand_pose'),
            jaw_pose_mean=self.mean_poses_dict.get('jaw_pose'),
        )
