import sys
from typing import Dict, Tuple, Optional
import os.path as osp

import pickle
import time

from loguru import logger
from collections import defaultdict
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit
from omegaconf import DictConfig

from .pose_utils import PoseParameterization
from .networks import build_regressor
from .keypoint_loss import KeypointLosses
from .rigid_alignment import RotationTranslationAlignment

from ..body_models import KeypointTensor
from ..backbone import build_backbone
from ..camera import CameraParams, build_cam_proj

from human_shape.losses import build_loss
from human_shape.data.structures import StructureList
from human_shape.utils import (
    Tensor, BlendShapeDescription, StringList, AppearanceDescription)


class AbstractRegressor(nn.Module):
    def __init__(
        self,
        body_model_cfg: DictConfig,
        network_cfg: DictConfig,
        loss_cfg: DictConfig,
        dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
        super(AbstractRegressor, self).__init__()

        # Pose only the final stage of the model to save computation
        self.pose_last_stage = network_cfg.get('pose_last_stage', True)
        logger.info(f'Pose last stage: {self.pose_last_stage}')

    @property
    def feat_dim(self) -> int:
        ''' Returns the dimension of the expected feature vector '''
        return self._feat_dim

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

        self.use_alignment = vertex_loss_cfg.get('use_alignment', False)
        if self.use_alignment:
            self.alignment = RotationTranslationAlignment()

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
        raise NotImplementedError

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
        raise NotImplementedError

    def forward(
        self,
        images: Tensor,
        targets: StructureList = None,
        compute_losses: bool = True,
        cond: Optional[Tensor] = None,
        extra_features: Optional[Tensor] = None,
        **kwargs
    ):
        raise NotImplementedError
