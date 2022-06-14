from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import time
from typing import Callable, Iterator, Union, Optional, List

import os.path as osp
import pickle
from loguru import logger

from functools import reduce

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .robustifiers import build_robustifier
from .utils import get_reduction_method
from human_shape.models.body_models import KeypointTensor

__all__ = ['GMofLoss',
           'KeypointLoss',
           'WeightedMSELoss',
           'WeightedL1Loss',
           'VertexEdgeLoss',
           'KeypointEdgeLoss',
           'SmoothL1LossModule',
           'RotationDistance',
           'build_loss',
           'build_adv_loss',
           ]


def GMof(residual, rho=1):
    squared_res = residual ** 2
    dist = torch.div(squared_res, squared_res + rho ** 2)
    return rho ** 2 * dist


def build_loss(type='l2', rho=100, reduction='mean', size_average=True,
               ignore_index=-100,
               **kwargs) -> nn.Module:
    logger.debug(f'Building loss: {type}')
    if type == 'gmof':
        return GMofLoss(rho=rho, reduction=reduction, **kwargs)
    elif type == 'keypoints':
        return KeypointLoss(reduction=reduction, **kwargs)
    elif type == 'l2':
        return WeightedMSELoss(reduction=reduction, **kwargs)
    elif type == 'weighted-l1':
        return WeightedL1Loss(
            reduction=reduction, size_average=size_average, **kwargs)
    elif type == 'keypoint-edge':
        return KeypointEdgeLoss(reduction=reduction, **kwargs)
    elif type == 'vertex-edge':
        return VertexEdgeLoss(reduction=reduction, **kwargs)
    elif type == 'bce':
        return nn.BCELoss()
    elif type == 'bce-logits':
        return nn.BCEWithLogitsLoss()
    elif type == 'cross-entropy':
        return nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index)
    elif type == 'l1':
        return nn.L1Loss()
    elif type == 'rotation':
        return RotationDistance(reduction=reduction, **kwargs)
    else:
        raise ValueError(f'Unknown loss type: {type}')


def build_adv_loss(discriminator, disc_cfg):
    adv_loss_type = disc_cfg.type
    weight = disc_cfg.weight

    if adv_loss_type == 'lsgan':
        return LSGANLoss(discriminator, weight=weight)
    elif adv_loss_type == 'wgan-gp':
        return WassersteinGANGP(discriminator, weight=weight,
                                **disc_cfg.wgan_gp)
    else:
        raise ValueError('Unknown adversarial loss: {}'.format(adv_loss_type))


class SmoothL1LossModule(nn.Module):
    def __init__(self, size_average=True, beta=1. / 9):
        super(SmoothL1LossModule, self).__init__()
        self.size_average = size_average
        self.beta = beta

    def extra_repr(self):
        return 'beta={}, size_average={}'.format(self.beta,
                                                 self.size_average)

    def forward(self, input, target):
        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2 / self.beta,
                           n - 0.5 * self.beta)
        if self.size_average:
            return loss.mean()
        return loss.sum()


class KeypointLoss(nn.Module):
    def __init__(
        self,
        norm_type='l1',
        binarize=False,
        robustifier=None,
        epsilon=1e-6,
        normalize='none',
        division='batch',
        **kwargs
    ) -> None:
        super(KeypointLoss, self).__init__()
        self.norm_type = norm_type
        assert self.norm_type in ['l1', 'l2'], 'Keypoint loss must be L1, L2'
        self.binarize = binarize
        self.robustifier = build_robustifier(
            robustifier_type=robustifier, **kwargs)
        self.epsilon = epsilon

        DIVISION_CHOICES = ['batch', 'visible']
        assert division in DIVISION_CHOICES, (
            f'division must be one of {DIVISION_CHOICES}, but got {division}'
        )
        self.division = division
        NORMALIZE_CHOICES = ['none', 'mean,std']
        assert normalize in NORMALIZE_CHOICES, (
            'normalize must be one of {NORMALIZE_CHOICES}, but got {normalize}'
        )
        self.normalize = normalize

    def extra_repr(self):
        msg = [
            f'Norm type: {self.norm_type.title()}',
            f'Division: {self.division}',
            f'Normalize: {self.normalize}',
            f'Epsilon: {self.epsilon}',
            f'Binarize: {self.binarize}',
        ]
        return '\n'.join(msg)

    def forward(self, input, target, weights=None, epsilon=1e-9):
        assert weights is not None
        keyp_dim = input.shape[-1]

        #  if self.binarize:
        #  weights = weights.gt(0).to(dtype=input.dtype)
        if self.normalize == 'none':
            norm_input = input
            norm_target = target
        elif self.normalize == 'mean,std':
            mean_gt = torch.mean(target, dim=1, keepdim=True)
            std_gt = torch.std(target, dim=1, keepdim=True)
            norm_input = (input - mean_gt) / (std_gt + epsilon)
            norm_target = (target - mean_gt) / (std_gt + epsilon)

        raw_diff = norm_input - norm_target
        # Should be B
        # Should contain the number of visible keypoints per batch item
        #  visibility = (weights.sum(dim=-1) * keyp_dim).view(-1, 1, 1)

        if self.robustifier is not None:
            diff = self.robustifier(raw_diff)
        else:
            if self.norm_type == 'l1':
                diff = raw_diff.abs()
            elif self.norm_type == 'l2':
                diff = raw_diff.pow(2)
        weighted_diff = diff * weights.unsqueeze(dim=-1)

        if self.division == 'batch':
            return torch.sum(weighted_diff) / weighted_diff.shape[0]
        elif self.division == 'visible':
            visibility = weights.gt(0)
            return torch.sum(weighted_diff) / (
                torch.sum(visibility) * 2.0 + epsilon)


class WeightedL1Loss(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(WeightedL1Loss, self).__init__()
        self.reduce_str = reduction
        self.reduce = get_reduction_method(reduction)

    def forward(self, input, target, weights=None):
        diff = input - target
        if weights is None:
            return diff.abs().sum() / diff.shape[0]
        else:
            diff = input - target
            weighted_diff = weights.unsqueeze(dim=-1) * diff.abs()
            return weighted_diff.sum() / diff.shape[0]


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(WeightedMSELoss, self).__init__()
        self.reduce_str = reduction
        self.reduce = get_reduction_method(reduction)

    def forward(self, input, target, weights=None):
        diff = input - target
        if weights is None:
            return diff.pow(2).sum() / diff.shape[0]
        else:
            return (
                weights.unsqueeze(dim=-1) * diff.pow(2)).sum() / diff.shape[0]


class GMofLoss(nn.Module):

    def __init__(self, rho=100, reduction='mean', **kwargs):
        super(GMofLoss, self).__init__()
        self.rho = rho
        self.reduction = get_reduction_method(reduction)
        self.reduction_str = reduction

    def extra_repr(self):
        return 'rho={}, reduction={}'.format(self.rho,
                                             self.reduction_str)

    def forward(self, module_input, target, weights=None):
        batch_size = module_input.shape[0]
        squared_residual = (module_input - target).pow(2)
        dist = torch.div(squared_residual, squared_residual + self.rho ** 2)
        output = self.rho ** 2 * dist
        if weights is not None:
            output *= weights.view(batch_size, -1, 1).pow(2)

        return self.reduction(output)


class LSGANLoss(nn.Module):
    def __init__(self, discriminator, weight=1.0):
        super(LSGANLoss, self).__init__()
        self.discriminator = discriminator
        self.weight = weight

    def extra_repr(self):
        return 'Weight: {}'.format(self.weight)

    def forward(self, fake, real=None, update_gen=True):
        if update_gen:
            fake_scores = self.discriminator(fake)

            fake_loss = torch.mean((fake_scores - 1).pow(2).sum(dim=1))
            return fake_loss * self.weight, {'fake_scores': fake_scores}
        else:
            all_scores = self.discriminator(torch.cat([fake, real], dim=0))
            fake_scores, real_scores = torch.split(all_scores, fake.shape[0],
                                                   dim=0)

            loss_real = torch.mean((real_scores - 1).pow(2).sum(dim=1))
            loss_fake = torch.mean(fake_scores.pow(2).sum(dim=1))

            return (loss_real + loss_fake) * self.weight, {
                'real_scores': real_scores}


class WassersteinGANGP(nn.Module):
    def __init__(self, discriminator, weight=1.0, gp_weight=1.0, gamma=1.0):
        super(WassersteinGANGP, self).__init__()
        self.discriminator = discriminator
        self.weight = weight
        self.gp_weight = gp_weight
        self.gamma = gamma

    def extra_repr(self):
        msg = 'Weight: {}\n'.format(self.weight)
        msg += 'GP Weight: {}\n'.format(self.gp_weight)
        msg += 'Lipschitz constant: {}\n'.format(self.gamma)
        return msg

    def forward(self, fake, real=None, update_gen=True):
        if update_gen:
            fake_score = self.discriminator(fake)
            return -torch.mean(fake_score), {}
        else:
            all_scores = self.discriminator(torch.cat([fake, real], dim=0))
            fake_scores, real_scores = torch.split(all_scores, fake.shape[0],
                                                   dim=0)

            loss_fake = torch.mean(fake_scores)
            loss_real = torch.mean(real_scores)

            loss = loss_fake - loss_real

            gp = self.compute_gp(fake, real)
            return loss + gp, {
                'wasserstein_distance': loss_real - loss_fake,
                'gp': gp}

    def compute_gp(self, fake, real):
        batch_size = real.shape[0]
        dtype = fake.dtype
        device = fake.device

        num_dims = real.dim() - 1
        tau = torch.zeros(
            [batch_size] + [1] * num_dims, requires_grad=True).to(
            dtype=dtype, device=device)
        tau.uniform_()

        x_tilde = tau * real + (1 - tau) * fake

        disc_x_tilde = self.discriminator(x_tilde)

        grad_out = torch.ones_like(disc_x_tilde)

        gradients = autograd.grad(outputs=disc_x_tilde,
                                  inputs=x_tilde,
                                  grad_outputs=grad_out,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        # Reshape the gradient tensor so that we get a long vector for every
        # element in the batch
        gradients = gradients.contiguous().view(batch_size, -1)
        penalty = ((gradients.norm(2, dim=1) - self.gamma) ** 2).mean(dim=0) \
            / self.gamma ** 2
        return penalty


class RotationDistance(nn.Module):
    def __init__(self, reduction='mean', epsilon=1e-7,
                 robustifier='none',
                 **kwargs):
        super(RotationDistance, self).__init__()
        self.reduction = get_reduction_method(reduction)
        self.reduction_str = reduction
        self.epsilon = epsilon
        self.robustifier = build_robustifier(
            robustifier_type=robustifier, epsilon=epsilon, **kwargs)

    def extra_repr(self) -> str:
        msg = []
        msg.append(f'Reduction: {self.reduction_str}')
        msg.append(f'Epsilon: {self.epsilon}')
        return '\n'.join(msg)

    def forward(self, module_input, target, weights=None):
        tr = torch.einsum(
            'bij,bij->b',
            [module_input.view(-1, 3, 3),
             target.view(-1, 3, 3)])

        theta = (tr - 1) * 0.5
        loss = torch.acos(
            torch.clamp(theta, -1 + self.epsilon, 1 - self.epsilon))
        if self.robustifier is not None:
            loss = self.robustifier(loss)
        if weights is not None:
            loss = loss.view(
                module_input.shape[0], -1) * weights.view(
                    module_input.shape[0], -1)
            return loss.sum() / (
                weights.gt(0).to(loss.dtype).sum() + self.epsilon)
        else:
            return loss.sum() / module_input.shape[0]


class VertexEdgeLoss(nn.Module):
    def __init__(self, norm_type='l2',
                 gt_edge_path='',
                 est_edge_path='',
                 robustifier=None,
                 edge_thresh=0.0, epsilon=1e-8, **kwargs):
        super(VertexEdgeLoss, self).__init__()

        assert norm_type in ['l1', 'l2'], 'Norm type must be [l1, l2]'
        self.norm_type = norm_type
        self.epsilon = epsilon
        self.robustifier = build_robustifier(
            robustifier_type=robustifier, **kwargs)

        gt_edge_path = osp.expandvars(gt_edge_path)
        est_edge_path = osp.expandvars(est_edge_path)
        self.has_connections = osp.exists(gt_edge_path) and osp.exists(
            est_edge_path)
        if self.has_connections:
            gt_edges = np.load(gt_edge_path)
            est_edges = np.load(est_edge_path)

            self.register_buffer(
                'gt_connections', torch.tensor(gt_edges, dtype=torch.long))
            self.register_buffer(
                'est_connections', torch.tensor(est_edges, dtype=torch.long))

    def extra_repr(self):
        msg = [
            f'Norm type: {self.norm_type}',
        ]
        if self.has_connections:
            msg.append(
                f'GT Connections shape: {self.gt_connections.shape}'
            )
            msg.append(
                f'Est Connections shape: {self.est_connections.shape}'
            )
        return '\n'.join(msg)

    def compute_edges(self, points, connections):
        start = torch.index_select(
            points, 1, connections[:, 0])
        end = torch.index_select(points, 1, connections[:, 1])
        return start - end

    def forward(self, gt_vertices, est_vertices, weights=None):
        if not self.has_connections:
            return 0.0

        # Compute the edges for the ground truth keypoints and the model keypoints
        # Remove the confidence from the ground truth keypoints
        gt_edges = self.compute_edges(
            gt_vertices, connections=self.gt_connections)
        est_edges = self.compute_edges(
            est_vertices, connections=self.est_connections)

        raw_edge_diff = (gt_edges - est_edges)

        batch_size = gt_vertices.shape[0]
        if self.robustifier is not None:
            raise NotImplementedError
        else:
            if self.norm_type == 'l2':
                return (raw_edge_diff.pow(2).sum(dim=-1)).sum() / batch_size
            elif self.norm_type == 'l1':
                return (raw_edge_diff.pow(2).sum(dim=-1)).sum() / batch_size
            else:
                raise NotImplementedError(
                    f'Loss type not implemented: {self.loss_type}')


class KeypointEdgeLoss(nn.Module):
    def __init__(self, norm_type='l2', connections=None,
                 robustifier=None,
                 edge_thresh=0.0, epsilon=1e-8, **kwargs):
        super(KeypointEdgeLoss, self).__init__()
        self.connections = None
        if connections is not None:
            connections = torch.tensor(connections).reshape(-1, 2)
            self.register_buffer('connections', connections)
        self.edge_thresh = edge_thresh

        assert norm_type in ['l1', 'l2'], 'Norm type must be [l1, l2]'
        self.norm_type = norm_type
        self.epsilon = epsilon
        self.robustifier = build_robustifier(
            robustifier_type=robustifier, **kwargs)

    def extra_repr(self):
        msg = [
            f'Edge threshold: {self.edge_thresh}',
            f'Norm type: {self.norm_type}',
        ]
        if self.connections is not None:
            msg.append(f'Connections shape: {self.connections.shape}')
        return '\n'.join(msg)

    def compute_edges(self, keypoints, connections):
        start = torch.index_select(
            keypoints, 1, connections[:, 0])
        end = torch.index_select(keypoints, 1, connections[:, 1])
        if isinstance(start, (KeypointTensor,)):
            start = start._t
        if isinstance(end, (KeypointTensor,)):
            end = end._t
        return start - end

    def forward(self, gt_keypoints, model_keypoints,
                connections=None,
                weights=None):
        if ((self.connections is None or len(self.connections) < 1) and
                (connections is None or len(connections) < 1)):
            return 0.0

        if connections is None and self.connections is not None:
            connections = self.connections

        dtype, device = gt_keypoints.dtype, gt_keypoints.device
        if not torch.is_tensor(connections):
            connections = torch.tensor(
                connections, dtype=torch.long, device=device).reshape(-1, 2)
        # Compute the edges for the ground truth keypoints and the model
        #  keypoints. Remove the confidence from the ground truth keypoints
        gt_edges = self.compute_edges(gt_keypoints, connections)
        model_edges = self.compute_edges(model_keypoints, connections)

        # Compute the confidence of the edge as the harmonic mean of the
        # confidences
        # Weights: BxC
        if weights is not None:
            weight_start_pt = torch.index_select(weights, 1, connections[:, 0])
            weight_end_pt = torch.index_select(weights, 1, connections[:, 1])
            edge_weight = 2.0 * weight_start_pt * weight_end_pt / (
                weight_start_pt + weight_end_pt + self.epsilon)
            edge_weight[torch.isnan(edge_weight)] = 0
        else:
            edge_weight = torch.ones_like(gt_edges[:, :, 0])

        raw_edge_diff = (gt_edges - model_edges)

        if self.robustifier is not None:
            raise NotImplementedError
        else:
            if self.norm_type == 'l2':
                return (raw_edge_diff.pow(2).sum(dim=-1) *
                        edge_weight).sum() / gt_keypoints.shape[0]
            elif self.norm_type == 'l1':
                return (raw_edge_diff.abs().sum(dim=-1) *
                        edge_weight).sum() / gt_keypoints.shape[0]
            else:
                raise NotImplementedError(
                    f'Loss type not implemented: {self.norm_type}')
