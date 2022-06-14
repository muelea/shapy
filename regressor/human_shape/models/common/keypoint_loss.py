from typing import Dict, List, Optional
import sys
import numpy as np

import torch
import torch.nn as nn

from loguru import logger

from human_shape.losses import build_loss
from human_shape.models.body_models.utils import KeypointTensor
from human_shape.utils import Tensor, StringList


class KeypointLosses(nn.Module):
    def __init__(self, joints_2d_cfg, joints_3d_cfg, edge_2d_cfg,
                 edge_3d_cfg, weights=None,
                 ):
        super(KeypointLosses, self).__init__()
        self.joints_2d_weight = joints_2d_cfg.weight
        if self.joints_2d_weight > 0:
            self.joints_2d_loss = build_loss(**joints_2d_cfg)
            logger.debug('2D joints weight, loss: {}',
                         self.joints_2d_weight, self.joints_2d_loss)

        self.joints_3d_weight = joints_3d_cfg.weight
        if self.joints_3d_weight > 0:
            self.joints_3d_loss = build_loss(**joints_3d_cfg)
            logger.debug('3D joints weight, loss: {}',
                         self.joints_3d_weight, self.joints_3d_loss)

        self.edge_2d_weight = edge_2d_cfg.weight
        if self.edge_2d_weight > 0:
            self.edge_2d_enable_at = edge_2d_cfg.enable
            self.edge_2d_loss = build_loss(**edge_2d_cfg)
            logger.debug('2D edge weight, loss: {}',
                         self.edge_2d_weight, self.edge_2d_loss)
            self.edge_2d_active = True

        self.edge_3d_weight = edge_3d_cfg.weight
        if self.edge_3d_weight > 0:
            self.edge_3d_enable_at = edge_3d_cfg.enable
            self.edge_3d_loss = build_loss(**edge_3d_cfg)
            logger.debug('3D edge weight, loss: {}',
                         self.edge_3d_weight, self.edge_3d_loss)
            self.edge_3d_active = True

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float32)
            self.register_buffer('weights', weights)
        else:
            self.weights = None

    def extra_repr(self) -> str:
        msg = [
            f'Joints 2D: {self.joints_2d_weight}',
            f'Joints 3D: {self.joints_3d_weight}',
            f'Edge 2D: {self.edge_2d_weight}',
            f'Edge 3D: {self.edge_3d_weight}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        est_keypoints2d: KeypointTensor,
        gt_keypoints2d: Tensor,
        est_keypoints3d: Tensor,
        gt_keypoints3d: Optional[Tensor] = None,
        part: str = '',
        center_around: Optional[StringList] = None
    ) -> Dict[str, Tensor]:
        device, dtype = est_keypoints2d.device, est_keypoints2d.device
        losses = {}
        if not self.training:
            return losses

        if center_around is None:
            center_around = []

        indices = est_keypoints2d._part_indices[part]
        connections = est_keypoints2d._part_connections[part]
        # If training calculate 2D projection loss
        if self.joints_2d_weight > 0:
            curr_weights = gt_keypoints2d[:, indices, -1]
            if self.weights is not None:
                curr_weights *= self.weights[None, :]
            joints_2d_loss = (
                self.joints_2d_weight * self.joints_2d_loss(
                    est_keypoints2d[:, indices],
                    gt_keypoints2d[:, indices, :-1],
                    weights=curr_weights))
            losses.update(joints_2d_loss=joints_2d_loss)

        if self.edge_2d_weight > 0 and self.edge_2d_active:
            edge_2d_loss = (
                self.edge_2d_weight * self.edge_2d_loss(
                    est_keypoints2d,
                    gt_keypoints2d[:, :, :-1],
                    connections=connections,
                    weights=gt_keypoints2d[:, :, -1]))
            losses.update(edge_2d_loss=edge_2d_loss)

        #  If training calculate 3D joints loss
        if gt_keypoints3d is not None and len(gt_keypoints3d) > 0:
            conf = gt_keypoints3d[:, :, -1]
            if len(center_around) > 0:
                centering_indices = [est_keypoints2d.keypoint_names.index(name)
                                     for name in center_around if name in
                                     est_keypoints2d._keypoint_names]
                # Center the predictions using the estimated point
                est_center_point = est_keypoints3d[
                    :, centering_indices, :].mean(dim=1, keepdim=True)
                centered_pred_joints = est_keypoints3d - est_center_point

                gt_center_point = gt_keypoints3d[
                    :, centering_indices, :-1].mean(
                        dim=1, keepdim=True)
                centered_gt_joints = (
                    gt_keypoints3d[:, :, :-1] - gt_center_point)
            else:
                centered_pred_joints = est_keypoints3d
                conf = gt_keypoints3d[:, :, :-1]

                centered_gt_joints = gt_keypoints3d[:, :, :-1]

            if self.joints_3d_weight > 0:
                joints_3d_loss = (
                    self.joints_3d_weight * self.joints_3d_loss(
                        centered_pred_joints[:, indices],
                        centered_gt_joints[:, indices],
                        weights=conf[:, indices]))
                losses.update(joints_3d_loss=joints_3d_loss)

            if self.edge_3d_weight > 0:
                edge_3d_loss = (
                    self.edge_3d_weight * self.edge_3d_loss(
                        est_keypoints3d[:, :], gt_keypoints3d[:, :, :-1],
                        connections=connections,
                        weights=gt_keypoints3d[:, :, -1])
                )
                losses.update(edge_3d_loss)

        return losses
