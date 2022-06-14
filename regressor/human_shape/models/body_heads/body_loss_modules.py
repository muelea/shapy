import sys
import time
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict

from loguru import logger

from human_shape.losses import build_loss, build_prior
from human_shape.data.structures import StructureList

from human_shape.utils import Tensor, TensorList, IntList, StringList

PARAM_KEYS = ['betas', 'expression', 'global_rot', 'body_pose', 'hand_pose',
              'jaw_pose']


class SMPLLossModule(nn.Module):
    def __init__(self, loss_cfg):
        super(SMPLLossModule, self).__init__()
        self.stages_to_penalize = loss_cfg.get('stages_to_penalize', [-1])
        logger.info(f'Stages to penalize: {self.stages_to_penalize}')

        self.loss_enabled = defaultdict(lambda: True)
        self.loss_activ_step = {}

        shape_loss_cfg = loss_cfg.get('shape', {})
        self.shape_weight = shape_loss_cfg.get('weight', 0.0)
        if self.shape_weight > 0:
            self.shape_loss = build_loss(**shape_loss_cfg)
            self.loss_activ_step['shape'] = shape_loss_cfg.enable

        global_rot_cfg = loss_cfg.get('global_rot', {})
        self.global_rot_weight = global_rot_cfg.weight
        if self.global_rot_weight > 0:
            global_rot_loss_type = global_rot_cfg.type
            self.global_rot_loss_type = global_rot_loss_type
            self.global_rot_loss = build_loss(**global_rot_cfg)
            logger.debug(
                'Global pose weight, loss: {}, {}',
                self.global_rot_weight, self.global_rot_loss)
            self.loss_activ_step['global_rot'] = global_rot_cfg.enable

        body_pose_loss_cfg = loss_cfg.get('body_pose', {})
        self.body_pose_weight = body_pose_loss_cfg.weight
        if self.body_pose_weight > 0:
            body_pose_loss_type = body_pose_loss_cfg.type
            self.body_pose_loss_type = body_pose_loss_type
            self.body_pose_loss = build_loss(**body_pose_loss_cfg)
            logger.debug('Body pose weight, loss: {}, {}',
                         self.body_pose_weight,
                         self.body_pose_loss)
            self.loss_activ_step['body_pose'] = body_pose_loss_cfg.enable

    def is_active(self) -> bool:
        return any(self.loss_enabled.values())

    def toggle_losses(self, step) -> None:
        for key in self.loss_activ_step:
            self.loss_enabled[key] = step >= self.loss_activ_step[key]

    def extra_repr(self) -> str:
        msg = [
            f'Shape weight: {self.shape_weight}',
            f'Global pose weight: {self.global_rot_weight}',
            f'Body pose weight: {self.body_pose_weight}',
        ]
        return '\n'.join(msg)

    def single_loss_step(
        self,
        parameters: Tensor,
        target_params: Dict[str, Tensor],
        target_param_idxs: Dict[str, Tensor],
        device: torch.device = None,
        gt_confs: Optional[Tensor] = None,
        keypoint_part_indices: Optional[Union[TensorList, IntList]] = None,
        penalize_only_parts: bool = False,
    ) -> Dict[str, Tensor]:
        losses = defaultdict(
            lambda: torch.tensor(0, device=device, dtype=torch.float32))

        compute_shape_loss = (
            self.shape_weight > 0 and self.loss_enabled['betas'] and
            'betas' in parameters and
            'betas' in target_params and not penalize_only_parts
        )
        if compute_shape_loss:
            shape_common_dim = min(parameters['betas'].shape[-1],
                                   target_params['betas'].shape[-1])
            gt_shape_idxs = target_param_idxs['betas']
            losses['shape_loss'] = (
                self.shape_loss(
                    parameters['betas'][gt_shape_idxs, :shape_common_dim],
                    target_params['betas'][:, :shape_common_dim]) *
                self.shape_weight)

        compute_global_rot_loss = (
            self.global_rot_weight > 0 and self.loss_enabled['betas'] and
            'global_rot' in target_params and not penalize_only_parts
        )
        if compute_global_rot_loss:
            global_rot_idxs = target_param_idxs['global_rot']
            losses['global_rot_loss'] = (
                self.global_rot_loss(
                    parameters['global_rot'][global_rot_idxs],
                    target_params['global_rot']) *
                self.global_rot_weight)

        compute_body_pose_loss = (
            self.body_pose_weight > 0 and self.loss_enabled['betas'] and
            'body_pose' in target_params and not penalize_only_parts)

        if compute_body_pose_loss:
            body_pose_idxs = target_param_idxs['body_pose']
            est_body_pose = parameters['body_pose'][body_pose_idxs]
            losses['body_pose_loss'] = (self.body_pose_loss(
                est_body_pose, target_params['body_pose']) *
                self.body_pose_weight)

        return losses

    def forward(
        self,
        network_params: Dict[str, Dict[str, Tensor]],
        targets: StructureList,
        gt_confs: Optional[Tensor] = None,
        keypoint_part_indices: Optional[Union[TensorList, IntList]] = None,
        device: torch.device = None,
    ) -> Dict[str, Tensor]:
        '''
        '''
        #  if device is None:
        #  device = next(self.parameters()).device

        start_idxs = defaultdict(lambda: 0)
        in_target_param_idxs = defaultdict(lambda: [])
        in_target_params = defaultdict(lambda: [])

        for idx, target in enumerate(targets):
            # If there are no 3D annotations, skip and add to the starting
            # index the number of bounding boxes
            if len(target) < 1:
                continue

            for param_key in PARAM_KEYS:
                if not target.has_field(param_key):
                    start_idxs[param_key] += len(target)
                    continue
                end_idx = start_idxs[param_key] + 1
                in_target_param_idxs[param_key] += list(
                    range(start_idxs[param_key], end_idx))
                start_idxs[param_key] += 1

                in_target_params[param_key].append(
                    target.get_field(param_key))

        target_params = {}
        for key, val in in_target_params.items():
            if key == 'hand_pose':
                target_params['left_hand_pose'] = torch.stack([
                    t.left_hand_pose
                    for t in val])
                target_params['right_hand_pose'] = torch.stack([
                    t.right_hand_pose
                    for t in val])
            elif key == 'betas' or key == 'expression':
                data_lst = [getattr(t, key) for t in val]
                common_dim = min([len(v) for v in data_lst])
                target_params[key] = torch.stack(
                    [v.reshape(-1)[:common_dim] for v in data_lst])
            else:
                target_params[key] = torch.stack(
                    [getattr(t, key) for t in val])

        target_param_idxs = {}
        for key in in_target_param_idxs.keys():
            if key == 'hand_pose':
                target_param_idxs['left_hand_pose'] = torch.tensor(
                    np.asarray(in_target_param_idxs[key]),
                    device=device,
                    dtype=torch.long)
                target_param_idxs['right_hand_pose'] = target_param_idxs[
                    'left_hand_pose'].clone()
            else:
                target_param_idxs[key] = torch.tensor(
                    np.asarray(in_target_param_idxs[key]),
                    device=device,
                    dtype=torch.long)

        stages_to_penalize = self.stages_to_penalize.copy()
        output_losses = {}
        num_stages = network_params.get('num_stages', 1)
        for ii, curr_key in enumerate(stages_to_penalize):
            curr_params = network_params[curr_key]
            if curr_params is None:
                logger.warning(f'Network output for {curr_key} is None')
                continue

            curr_losses = self.single_loss_step(
                curr_params, target_params,
                target_param_idxs, device=device,
                gt_confs=gt_confs,
                keypoint_part_indices=keypoint_part_indices,
            )
            for key in curr_losses:
                output_losses[f'{curr_key}_{key}'] = curr_losses[key]

        return output_losses


class SMPLHLossModule(SMPLLossModule):
    def __init__(self, loss_cfg):
        super(SMPLHLossModule, self).__init__(loss_cfg)

        left_hand_pose_cfg = loss_cfg.get('left_hand_pose', {})
        left_hand_pose_loss_type = loss_cfg.left_hand_pose.type
        self.lhand_use_conf = left_hand_pose_cfg.get('use_conf_weight', False)

        self.left_hand_pose_weight = loss_cfg.left_hand_pose.weight
        if self.left_hand_pose_weight > 0:
            self.left_hand_pose_loss_type = left_hand_pose_loss_type
            self.left_hand_pose_loss = build_loss(**loss_cfg.left_hand_pose)
            self.loss_activ_step[
                'left_hand_pose'] = loss_cfg.left_hand_pose.enable
            logger.debug('Left hand pose weight, loss: {}, {}',
                         self.left_hand_pose_weight, self.left_hand_pose_loss)

        right_hand_pose_cfg = loss_cfg.get('right_hand_pose', {})
        right_hand_pose_loss_type = loss_cfg.right_hand_pose.type
        self.right_hand_pose_weight = loss_cfg.right_hand_pose.weight
        if self.right_hand_pose_weight > 0:
            self.rhand_use_conf = right_hand_pose_cfg.get(
                'use_conf_weight', False)
            self.right_hand_pose_loss_type = right_hand_pose_loss_type
            self.right_hand_pose_loss = build_loss(**loss_cfg.right_hand_pose)
            self.loss_activ_step[
                'right_hand_pose'] = loss_cfg.right_hand_pose.enable
            logger.debug('Right hand pose weight, loss: {}, {}',
                         self.right_hand_pose_weight,
                         self.right_hand_pose_loss)

    def extra_repr(self) -> str:
        desc = super(SMPLHLossModule, self).extra_repr()
        msg = [
            desc,
            f'Left hand pose weight: {self.left_hand_pose_weight}',
            f'Right hand pose weight {self.right_hand_pose_weight}',
        ]
        return '\n'.join(msg)

    def single_loss_step(
        self,
        parameters: Tensor,
        target_params: Dict[str, Tensor],
        target_param_idxs: Dict[str, Tensor],
        device: torch.device = None,
        gt_confs: Optional[Tensor] = None,
        keypoint_part_indices: Optional[Union[TensorList, IntList]] = None,
        penalize_only_parts: bool = False,
    ) -> Dict[str, Tensor]:
        losses = super(SMPLHLossModule, self).single_loss_step(
            parameters, target_params,
            target_param_idxs,
            device=device,
            gt_confs=gt_confs,
            keypoint_part_indices=keypoint_part_indices,
            penalize_only_parts=penalize_only_parts,
        )

        if (self.left_hand_pose_weight > 0 and
                self.loss_enabled['left_hand_pose'] and
                'left_hand_pose' in parameters and
                'left_hand_pose' in target_param_idxs):
            est_left_hand_pose = parameters['left_hand_pose']
            # Get the batch size and the number of joints
            batch_size, num_left_hand_joints = est_left_hand_pose.shape[:2]
            # Get the indices of the keypoints of the left hand
            left_hand_indices = keypoint_part_indices['left_hand']
            # Convert to a tensor, if it is not already one
            if not torch.is_tensor(left_hand_indices):
                left_hand_indices = torch.tensor(
                    left_hand_indices, device=device, dtype=torch.long)
            # Get the indices of the targets that have ground-truth left hand
            # pose parameters
            has_left_hand = target_param_idxs['left_hand_pose']
            # Get the confidence scores of the left hand
            if self.lhand_use_conf:
                left_hand_conf = gt_confs[:, left_hand_indices]
                weights = left_hand_conf.mean(axis=1, keepdim=True).expand(
                    -1, num_left_hand_joints).reshape(-1)
                weights = weights.view(-1, num_left_hand_joints)
            else:
                weights = torch.ones(
                    [batch_size, num_left_hand_joints],
                    dtype=est_left_hand_pose.dtype,
                    device=est_left_hand_pose.device)
            # Keep the weights for the targets that have ground-truth
            weights = weights[has_left_hand]
            losses['left_hand_pose_loss'] = (
                self.left_hand_pose_loss(
                    parameters['left_hand_pose'][has_left_hand],
                    target_params['left_hand_pose'], weights=weights) *
                self.left_hand_pose_weight)

        if (self.right_hand_pose_weight > 0 and
                self.loss_enabled['right_hand_pose'] and
                'right_hand_pose' in parameters and
                'right_hand_pose' in target_param_idxs):
            est_right_hand_pose = parameters['right_hand_pose']
            # Get the batch size and the number of joints
            batch_size, num_right_hand_joints = est_right_hand_pose.shape[:2]
            # Get the indices of the keypoints of the right hand
            right_hand_indices = keypoint_part_indices['right_hand']
            # Convert to a tensor, if it is not already one
            if not torch.is_tensor(right_hand_indices):
                right_hand_indices = torch.tensor(
                    right_hand_indices, device=device, dtype=torch.long)
            # Get the indices of the targets that have ground-truth right hand
            # pose parameters
            has_right_hand = target_param_idxs['right_hand_pose']
            # Get the confidence scores of the right hand
            right_hand_conf = gt_confs[:, right_hand_indices]
            if self.lhand_use_conf:
                weights = right_hand_conf.mean(axis=1, keepdim=True).expand(
                    -1, num_right_hand_joints).reshape(-1)
                weights = weights.view(-1, num_right_hand_joints)
            else:
                weights = torch.ones(
                    [batch_size, num_right_hand_joints],
                    dtype=est_right_hand_pose.dtype,
                    device=est_right_hand_pose.device)
            # Keep the weights for the targets that have ground-truth
            weights = weights[has_right_hand]
            losses['right_hand_pose_loss'] = (
                self.right_hand_pose_loss(
                    parameters['right_hand_pose'][has_right_hand],
                    target_params['right_hand_pose'], weights=weights) *
                self.right_hand_pose_weight)

        return losses


class SMPLXLossModule(SMPLHLossModule):
    '''
    '''

    def __init__(self, loss_cfg):
        super(SMPLXLossModule, self).__init__(loss_cfg)

        expression_cfg = loss_cfg.get('expression', {})
        self.expr_use_conf_weight = expression_cfg.get(
            'use_conf_weight', False)
        self.expression_weight = expression_cfg.weight
        if self.expression_weight > 0:
            self.expression_loss = build_loss(**expression_cfg)
            self.loss_activ_step['expression'] = expression_cfg.enable
            logger.debug('Expression weight, loss: {}, {}',
                         self.expression_weight, self.expression_loss)

        jaw_pose_cfg = loss_cfg.get('jaw_pose', {})
        self.jaw_pose_weight = jaw_pose_cfg.get('weight', 0.0)
        self.jaw_use_conf_weight = jaw_pose_cfg.get('use_conf_weight', False)
        if self.jaw_pose_weight > 0:
            jaw_pose_loss_type = jaw_pose_cfg.type
            self.jaw_pose_loss_type = jaw_pose_loss_type
            self.jaw_pose_loss = build_loss(**loss_cfg.jaw_pose)
            self.loss_activ_step['jaw_pose'] = loss_cfg.jaw_pose.enable
            logger.debug('Jaw pose weight, loss: {}, {}',
                         self.jaw_pose_weight, self.jaw_pose_loss)

    def extra_repr(self) -> str:
        desc = super(SMPLXLossModule, self).extra_repr()
        msg = [
            desc,
            f'Expression weight: {self.expression_weight}',
            f'Jaw pose prior weight: {self.jaw_pose_weight}',
        ]
        return '\n'.join(msg)

    def single_loss_step(
        self,
        parameters: Tensor,
        target_params: Dict[str, Tensor],
        target_param_idxs: Dict[str, Tensor],
        device: torch.device = None,
        gt_confs: Optional[Tensor] = None,
        keypoint_part_indices: Optional[Union[TensorList, IntList]] = None,
        penalize_only_parts: bool = False,
    ) -> Dict[str, Tensor]:
        losses = super(SMPLXLossModule, self).single_loss_step(
            parameters, target_params,
            target_param_idxs,
            device=device,
            gt_confs=gt_confs,
            keypoint_part_indices=keypoint_part_indices,
            penalize_only_parts=penalize_only_parts,
        )

        compute_expr_loss = (self.expression_weight > 0 and
                             self.loss_enabled['expression'] and
                             'expression' in parameters and
                             'expression' in target_param_idxs)
        if compute_expr_loss:
            est_expression = parameters['expression']
            batch_size = len(est_expression)
            has_expression = target_param_idxs['expression']
            num_ones = [1]
            if self.expr_use_conf_weight:
                face_indices = keypoint_part_indices['face']
                face_conf = gt_confs[:, face_indices]
                weights = face_conf.mean(axis=1)
                weights = weights.view(-1)
            else:
                weights = torch.ones([batch_size],
                                     dtype=est_expression.dtype,
                                     device=est_expression.device)

            weights = weights[has_expression]

            expr_common_dim = min(
                parameters['expression'].shape[-1],
                target_params['expression'].shape[-1])

            losses['expression_loss'] = (
                self.expression_loss(
                    est_expression[has_expression, :expr_common_dim],
                    target_params['expression'][:, :expr_common_dim],
                    weights=weights) *
                self.expression_weight)

        if (self.jaw_pose_weight > 0 and self.loss_enabled['jaw_pose'] and
                'jaw_pose' in parameters and
                'jaw_pose' in target_param_idxs):
            has_jaw_pose = target_param_idxs['jaw_pose']
            est_jaw_pose = parameters['jaw_pose']
            batch_size = est_jaw_pose.shape[0]
            num_ones = [1]
            if self.jaw_use_conf_weight:
                face_indices = keypoint_part_indices['face']
                face_conf = gt_confs[:, face_indices]
                weights = face_conf.mean(axis=1)
                weights = weights.view(-1, *num_ones)
            else:
                weights = torch.ones([batch_size],
                                     dtype=est_jaw_pose.dtype,
                                     device=est_jaw_pose.device)
            weights = weights[has_jaw_pose]

            losses['jaw_pose_loss'] = (
                self.jaw_pose_loss(
                    est_jaw_pose[has_jaw_pose],
                    target_params['jaw_pose'],
                    weights=weights) * self.jaw_pose_weight)

        return losses


class RegularizerModule(nn.Module):
    def __init__(
        self, loss_cfg,
        body_pose_mean: Optional[Tensor] = None,
        left_hand_pose_mean: Optional[Tensor] = None,
        right_hand_pose_mean: Optional[Tensor] = None,
        jaw_pose_mean: Optional[Tensor] = None
    ) -> None:
        ''' SMPL/SMPL+H/SMPL-X parameter regularizer
        '''
        super(RegularizerModule, self).__init__()

        self.stages_to_regularize = loss_cfg.get(
            'stages_to_penalize', [])
        logger.info(f'Stages to regularize: {self.stages_to_regularize}')

        # Construct the shape prior
        shape_prior_type = loss_cfg.shape.prior.type
        self.shape_prior_weight = loss_cfg.shape.prior.weight
        if self.shape_prior_weight > 0:
            self.shape_prior = build_prior(shape_prior_type,
                                           **loss_cfg.shape.prior)
            logger.debug(f'Shape prior {self.shape_prior}')

        # Construct the expression prior
        expression_prior_cfg = loss_cfg.expression.prior
        expression_prior_type = expression_prior_cfg.type
        self.expression_prior_weight = expression_prior_cfg.weight
        if self.expression_prior_weight > 0:
            self.expression_prior = build_prior(
                expression_prior_type,
                **expression_prior_cfg)
            logger.debug(f'Expression prior {self.expression_prior}')

        # Construct the body pose prior
        body_pose_prior_cfg = loss_cfg.body_pose.prior
        body_pose_prior_type = body_pose_prior_cfg.type
        self.body_pose_prior_weight = body_pose_prior_cfg.weight
        if self.body_pose_prior_weight > 0:
            self.body_pose_prior = build_prior(
                body_pose_prior_type,
                mean=body_pose_mean,
                **body_pose_prior_cfg)
            logger.debug(f'Body pose prior {self.body_pose_prior}')

        # Construct the left hand pose prior
        left_hand_prior_cfg = loss_cfg.left_hand_pose.prior
        left_hand_pose_prior_type = left_hand_prior_cfg.type
        self.left_hand_pose_prior_weight = left_hand_prior_cfg.weight
        if self.left_hand_pose_prior_weight > 0:
            self.left_hand_pose_prior = build_prior(
                left_hand_pose_prior_type,
                mean=left_hand_pose_mean,
                **left_hand_prior_cfg)
            logger.debug(f'Left hand pose prior {self.left_hand_pose_prior}')

        # Construct the right hand pose prior
        right_hand_prior_cfg = loss_cfg.right_hand_pose.prior
        right_hand_pose_prior_type = right_hand_prior_cfg.type
        self.right_hand_pose_prior_weight = right_hand_prior_cfg.weight
        if self.right_hand_pose_prior_weight > 0:
            self.right_hand_pose_prior = build_prior(
                right_hand_pose_prior_type, mean=right_hand_pose_mean,
                **right_hand_prior_cfg)
            logger.debug(f'Right hand pose prior {self.right_hand_pose_prior}')

        # Construct the jaw pose prior
        jaw_pose_prior_cfg = loss_cfg.jaw_pose.prior
        jaw_pose_prior_type = jaw_pose_prior_cfg.type
        self.jaw_pose_prior_weight = jaw_pose_prior_cfg.weight
        if self.jaw_pose_prior_weight > 0:
            self.jaw_pose_prior = build_prior(
                jaw_pose_prior_type, mean=jaw_pose_mean, **jaw_pose_prior_cfg)
            logger.debug(f'Jaw pose prior {self.jaw_pose_prior}')

        logger.debug(self)

    def extra_repr(self) -> str:
        msg = []
        if self.shape_prior_weight > 0:
            msg.append('Shape prior weight: {}'.format(
                self.shape_prior_weight))
        if self.expression_prior_weight > 0:
            msg.append('Expression prior weight: {}'.format(
                self.expression_prior_weight))
        if self.body_pose_prior_weight > 0:
            msg.append('Body pose prior weight: {}'.format(
                self.body_pose_prior_weight))
        if self.left_hand_pose_prior_weight > 0:
            msg.append('Left hand pose prior weight: {}'.format(
                self.left_hand_pose_prior_weight))
        if self.right_hand_pose_prior_weight > 0:
            msg.append('Right hand pose prior weight {}'.format(
                self.right_hand_pose_prior_weight))
        if self.jaw_pose_prior_weight > 0:
            msg.append('Jaw pose prior weight: {}'.format(
                self.jaw_pose_prior_weight))
        return '\n'.join(msg)

    def single_regularization_step(
        self, parameters,
        penalize_only_parts=False,
        genders: Optional[StringList] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        prior_losses = {}

        betas = parameters.get('betas', None)
        reg_shape = (self.shape_prior_weight > 0 and betas is not None and
                     not penalize_only_parts)
        if reg_shape:
            prior_losses['shape_prior'] = (
                self.shape_prior_weight * self.shape_prior(
                    betas, genders=genders))

        expression = parameters.get('expression', None)
        reg_expression = (
            self.expression_prior_weight > 0 and expression is not None)
        if reg_expression:
            prior_losses['expression_prior'] = (
                self.expression_prior(expression) *
                self.expression_prior_weight)

        body_pose = parameters.get('body_pose', None)
        betas = parameters.get('betas', None)
        reg_body_pose = (
            self.body_pose_prior_weight > 0 and body_pose is not None and
            not penalize_only_parts)
        if reg_body_pose:
            prior_losses['body_pose_prior'] = (
                self.body_pose_prior(body_pose) *
                self.body_pose_prior_weight)

        left_hand_pose = parameters.get('left_hand_pose', None)
        if (self.left_hand_pose_prior_weight > 0 and
                left_hand_pose is not None):
            prior_losses['left_hand_pose_prior'] = (
                self.left_hand_pose_prior(left_hand_pose) *
                self.left_hand_pose_prior_weight)

        right_hand_pose = parameters.get('right_hand_pose', None)
        if (self.right_hand_pose_prior_weight > 0 and
                right_hand_pose is not None):
            prior_losses['right_hand_pose_prior'] = (
                self.right_hand_pose_prior(right_hand_pose) *
                self.right_hand_pose_prior_weight)

        jaw_pose = parameters.get('jaw_pose', None)
        if self.jaw_pose_prior_weight > 0 and jaw_pose is not None:
            prior_losses['jaw_pose_prior'] = (
                self.jaw_pose_prior(jaw_pose) *
                self.jaw_pose_prior_weight)

        return prior_losses

    def forward(
        self,
        network_params: Dict,
        targets: Optional[StructureList] = None,
        **kwargs
    ) -> Dict[str, Tensor]:

        prior_losses = defaultdict(lambda: 0)
        if len(self.stages_to_regularize) < 1:
            return prior_losses

        genders = [t.get_field('gender') if t.has_field('gender') else ''
                   for t in targets]

        for ii, curr_key in enumerate(self.stages_to_regularize):
            curr_params = network_params.get(curr_key, None)
            if curr_params is None:
                logger.warning(f'Network output for {curr_key} is None')
                continue

            curr_losses = self.single_regularization_step(curr_params,
                                                          genders=genders)
            for key in curr_losses:
                prior_losses[f'{curr_key}_{key}'] = curr_losses[key]

        return prior_losses
