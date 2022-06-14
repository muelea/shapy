from typing import Dict, Tuple, Union, Optional

import sys
import os.path as osp
import pickle
from dataclasses import dataclass, fields

import math
import numpy as np

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F


from human_shape.utils import (batch_rodrigues, batch_rot2aa, Tensor)


def build_pose_decoder(cfg, num_angles, mean_pose=None, pca_basis=None):
    param_type = cfg.get('param_type', 'aa')
    logger.debug('Creating {} for {} joints', param_type, num_angles)
    if param_type == 'aa':
        decoder = AADecoder(num_angles=num_angles, mean=mean_pose, **cfg)
    elif param_type == 'pca':
        decoder = PCADecoder(pca_basis=pca_basis, mean=mean_pose, **cfg)
    elif param_type == 'cont_rot_repr':
        decoder = ContinuousRotReprDecoder(num_angles, mean=mean_pose, **cfg)
    elif param_type == 'rot_mats':
        decoder = SVDRotationProjection()
    else:
        raise ValueError(f'Unknown pose decoder: {param_type}')
    return decoder


class RotationMatrixRegressor(nn.Linear):

    def __init__(self, input_dim, num_angles, dtype=torch.float32,
                 append_params=True, **kwargs):
        super(RotationMatrixRegressor, self).__init__(
            input_dim + append_params * num_angles * 3,
            num_angles * 9)
        self.num_angles = num_angles
        self.dtype = dtype
        self.svd_projector = SVDRotationProjection()

    @property
    def name(self):
        return 'rotation'

    def get_param_dim(self):
        return 9

    def get_dim_size(self):
        return self.num_angles * 9

    def get_mean(self):
        return torch.eye(3, dtype=self.dtype).unsqueeze(dim=0).expand(
            self.num_angles, -1, -1)

    def forward(self, module_input):
        rot_mats = super(RotationMatrixRegressor, self).forward(
            module_input).view(-1, 3, 3)

        # Project the matrices on the manifold of rotation matrices using SVD
        rot_mats = self.svd_projector(rot_mats).view(
            -1, self.num_angles, 3, 3)

        return rot_mats


class ContinuousRotReprDecoder(nn.Module):
    ''' Decoder for transforming a latent representation to rotation matrices

        Implements the decoding method described in:
        "On the Continuity of Rotation Representations in Neural Networks"
    '''

    def __init__(self, num_angles, dtype=torch.float32, mean=None,
                 **kwargs):
        super(ContinuousRotReprDecoder, self).__init__()
        self.num_angles = num_angles
        self.dtype = dtype

        if isinstance(mean, dict):
            mean = mean.get('cont_rot_repr', None)
        if mean is None:
            mean = torch.tensor(
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                dtype=self.dtype).unsqueeze(dim=0).expand(
                    self.num_angles, -1).contiguous().view(-1)

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        mean = mean.reshape(-1, 6)

        if mean.shape[0] < self.num_angles:
            logger.debug(mean.shape)
            mean = mean.repeat(
                self.num_angles // mean.shape[0] + 1, 1).contiguous()
            mean = mean[:self.num_angles]
        elif mean.shape[0] > self.num_angles:
            mean = mean[:self.num_angles]

        mean = mean.reshape(-1)
        self.register_buffer('mean', mean)

    def get_type(self):
        return 'cont_rot_repr'

    def extra_repr(self):
        msg = 'Num angles: {}\n'.format(self.num_angles)
        msg += 'Mean: {}'.format(self.mean.shape)
        return msg

    def get_param_dim(self):
        return 6

    def get_dim_size(self):
        return self.num_angles * 6

    def get_mean(self):
        return self.mean.clone()

    def to_offsets(self, x):
        latent = x.reshape(-1, 3, 3)[:, :3, :2].reshape(-1, 6)
        return (latent - self.mean).reshape(x.shape[0], -1, 6)

    def encode(self, x, subtract_mean=False):
        orig_shape = x.shape
        if subtract_mean:
            raise NotImplementedError
        output = x.reshape(-1, 3, 3)[:, :3, :2].contiguous()
        return output.reshape(
            orig_shape[0], orig_shape[1], 3, 2)

    def forward(self, module_input):
        batch_size = module_input.shape[0]
        reshaped_input = module_input.view(-1, 3, 2)

        # Normalize the first vector
        b1 = F.normalize(reshaped_input[:, :, 0].clone(), dim=1)

        dot_prod = torch.sum(
            b1 * reshaped_input[:, :, 1].clone(), dim=1, keepdim=True)
        # Compute the second vector by finding the orthogonal complement to it
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=1)
        # Finish building the basis by taking the cross product
        b3 = torch.cross(b1, b2, dim=1)
        rot_mats = torch.stack([b1, b2, b3], dim=-1)

        return rot_mats.view(batch_size, -1, 3, 3)


class ContinuousRotReprRegressor(nn.Linear):
    def __init__(self, input_dim, num_angles, dtype=torch.float32,
                 append_params=True, **kwargs):
        super(ContinuousRotReprRegressor, self).__init__(
            input_dim + append_params * num_angles * 6, num_angles * 6)
        self.append_params = append_params
        self.num_angles = num_angles
        self.repr_decoder = ContinuousRotReprDecoder(num_angles)

    def get_dim_size(self):
        return self.num_angles * 9

    def get_mean(self):
        if self.to_aa:
            return torch.zeros([1, self.num_angles * 3], dtype=self.dtype)
        else:
            return torch.zeros([1, self.num_angles, 3, 3], dtype=self.dtype)

    def forward(self, module_input, prev_val):
        if self.append_params:
            if self.to_aa:
                prev_val = batch_rodrigues(prev_val)
            prev_val = prev_val[:, :, :2].contiguous().view(
                -1, self.num_angles * 6)

            module_input = torch.cat([module_input, prev_val], dim=-1)

        cont_repr = super(ContinuousRotReprRegressor,
                          self).forward(module_input)

        output = self.repr_decoder(cont_repr).view(-1, self.num_angles, 3, 3)
        return output


class SVDRotationProjection(nn.Module):
    def __init__(self, **kwargs):
        super(SVDRotationProjection, self).__init__()

    def forward(self, module_input):
        # Before converting the output rotation matrices of the VAE to
        # axis-angle representation, we first need to make them in to valid
        # rotation matrices
        with torch.no_grad():
            # TODO: Replace with Batch SVD once merged
            # Iterate over the batch dimension and compute the SVD
            svd_input = module_input.detach().cpu()
            #  svd_input = output
            norm_rotation = torch.zeros_like(svd_input)
            for bidx in range(module_input.shape[0]):
                U, _, V = torch.svd(svd_input[bidx])

                # Multiply the U, V matrices to get the closest orthonormal
                # matrix
                norm_rotation[bidx] = torch.matmul(U, V.t())
            norm_rotation = norm_rotation.to(module_input.device)

        # torch.svd supports backprop only for full-rank matrices.
        # The output is calculated as the valid rotation matrix plus the
        # output minus the detached output. If one writes down the
        # computational graph for this operation, it will become clear the
        # output is the desired valid rotation matrix, while for the
        # backward pass gradients are propagated only to the original
        # matrix
        # Source: PyTorch Gumbel-Softmax hard sampling
        # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
        correct_rot = norm_rotation - module_input.detach() + module_input
        return correct_rot


class AARegressor(nn.Linear):
    def __init__(self, input_dim, num_angles, dtype=torch.float32,
                 append_params=True, to_aa=True, **kwargs):
        super(AARegressor, self).__init__(
            input_dim + append_params * num_angles * 3, num_angles * 3)
        self.num_angles = num_angles
        self.to_aa = to_aa
        self.dtype = dtype

    def get_type(self):
        return 'aa'

    def get_param_dim(self):
        return 3

    def get_dim_size(self):
        return self.num_angles * 3

    def get_mean(self):
        return torch.zeros([self.num_angles * 3], dtype=self.dtype)

    def forward(self, features):
        aa_vectors = super(AARegressor, self).forward(features).view(
            -1, self.num_angles, 3)

        return batch_rodrigues(aa_vectors.view(-1, 3)).view(
            -1, self.num_angles, 3, 3)


class AADecoder(nn.Module):
    def __init__(self, num_angles, dtype=torch.float32, mean=None, **kwargs):
        super(AADecoder, self).__init__()
        self.num_angles = num_angles
        self.dtype = dtype

        if isinstance(mean, dict):
            mean = mean.get('aa', None)
        if mean is None:
            mean = torch.zeros([num_angles * 3], dtype=dtype)

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=dtype)
        mean = mean.reshape(-1)
        self.register_buffer('mean', mean)

    def get_dim_size(self):
        return self.num_angles * 3

    def get_mean(self):
        return torch.zeros([self.get_dim_size()], dtype=self.dtype)

    def forward(self, module_input):
        batch_size = module_input.shape[0]
        output = batch_rodrigues(module_input.view(-1, 3)).view(
            -1, self.num_angles, 3, 3)
        return output


class PCADecoder(nn.Module):
    def __init__(self, num_pca_comps=12, pca_basis=None, dtype=torch.float32,
                 mean=None,
                 **kwargs):
        super(PCADecoder, self).__init__()
        self.num_pca_comps = num_pca_comps
        self.dtype = dtype
        pca_basis_tensor = torch.tensor(pca_basis, dtype=dtype)
        self.register_buffer('pca_basis',
                             pca_basis_tensor[:self.num_pca_comps])
        inv_basis = torch.inverse(
            pca_basis_tensor.t()).unsqueeze(dim=0)
        self.register_buffer('inv_pca_basis', inv_basis)

        if isinstance(mean, dict):
            mean = mean.get('aa', None)

        if mean is None:
            mean = torch.zeros([45], dtype=dtype)

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=dtype)
        mean = mean.reshape(-1).reshape(1, -1)
        self.register_buffer('mean', mean)

    def get_type(self):
        return 'pca'

    def get_param_dim(self):
        return self.num_pca_comps

    def extra_repr(self):
        msg = 'PCA Components = {}'.format(self.num_pca_comps)
        return msg

    def get_mean(self):
        return self.mean.clone()

    def get_dim_size(self):
        return self.num_pca_comps

    def to_offsets(self, x):
        batch_size = x.shape[0]
        # Convert the rotation matrices to axis angle
        aa = batch_rot2aa(x.reshape(-1, 3, 3)).reshape(batch_size, 45, 1)

        # Project to the PCA space
        offsets = torch.matmul(
            self.inv_pca_basis, aa
        ).reshape(batch_size, -1)[:, :self.num_pca_comps]
        # Remove the mean offset
        #  offsets -= self.mean

        return offsets - self.mean

    def encode(self, x, subtract_mean=False):
        batch_size = x.shape[0]
        # Convert the rotation matrices to axis angle
        aa = batch_rot2aa(x.reshape(-1, 3, 3)).reshape(batch_size, 45, 1)

        # Project to the PCA space
        output = torch.matmul(
            self.inv_pca_basis, aa
        ).reshape(batch_size, -1)[:, :self.num_pca_comps]
        if subtract_mean:
            # Remove the mean offset
            output -= self.mean

        return output

    def forward(self, pca_coeffs):
        batch_size = pca_coeffs.shape[0]
        decoded = torch.einsum(
            'bi,ij->bj', [pca_coeffs, self.pca_basis]) + self.mean

        return batch_rodrigues(decoded.view(-1, 3)).view(
            batch_size, -1, 3, 3)


class EulerDecoder(nn.Module):
    def __init__(self, num_angles, euler_order='xyz',
                 dtype=torch.float32, **kwargs):
        super(EulerDecoder, self).__init__()
        self.euler_order = euler_order

    def get_type(self):
        return 'euler'

    def forward(self, euler_angles):
        ''' Forward operatior for the Euler angle decoder

            Parameters
            ----------
            euler_angles: torch.tensor, dtype=torch.float32
                A tensor with size BxJx3, where J is the number of angles and
                the last dimension corresponds to the 3 euler angles
            Returns
            -------
            rot_mats: torch.tensor, dtype=torch.float32

        '''
        batch_size, num_joints = euler_angles.shape[:2]

        angle1, angle2, angle3 = torch.chunk(euler_angles, 3, dim=-1)
        rot_mats1 = self.angle_to_rot_mat(angle1, axis=self.euler_order[0])
        rot_mats2 = self.angle_to_rot_mat(angle2, axis=self.euler_order[1])
        rot_mats3 = self.angle_to_rot_mat(angle3, axis=self.euler_order[2])

        # Should be (B*J)x3x3
        rot_mats = rot_mats1 @ rot_mats2 @ rot_mats3
        return rot_mats.view(batch_size, num_joints, 3, 3)

    def get_param_dim(self):
        return 3

    def angle_to_rot_mat(self, angle, axis='x'):
        cos_theta = torch.cos(angle).view(-1)
        sin_theta = torch.sin(angle).view(-1)

        zeros = torch.zeros_like(cos_theta)
        ones = torch.ones_like(cos_theta)
        if axis == 'x':
            rot_mats = torch.stack(
                [ones, zeros, zeros,
                 zeros, cos_theta, -sin_theta,
                 zeros, sin_theta, cos_theta], dim=1).view(-1, 3, 3)
        elif axis == 'y':
            rot_mats = torch.stack(
                [cos_theta, zeros, sin_theta,
                 zeros, ones, zeros,
                 -sin_theta, zeros, cos_theta], dim=1).view(-1, 3, 3)
        elif axis == 'z':
            rot_mats = torch.stack(
                [cos_theta, -sin_theta, zeros,
                 sin_theta, cos_theta, zeros,
                 zeros, zeros, ones], dim=1).view(-1, 3, 3)
        else:
            raise ValueError('Wrong axis value {}'.format(axis))
        return rot_mats


@dataclass
class PoseParameterization:
    dim: int
    ind_dim: int
    decoder: Optional[
        Union[AADecoder, ContinuousRotReprDecoder, SVDRotationProjection]
    ] = None
    mean: Optional[Tensor] = None
    regressor: Optional[Union[AARegressor, ContinuousRotReprRegressor,
                              RotationMatrixRegressor]] = None

    def keys(self):
        return [f.name for f in fields(PoseParameterization)]
        #  return [key for key in self.KEYS if getattr(self, key) is not None]

    def __getitem__(self, key):
        return getattr(self, key)


def build_pose_parameterization(
    num_angles,
    type='aa',
    num_pca_comps=12,
    latent_dim_size=32,
    append_params=True,
    **kwargs
) -> PoseParameterization:

    logger.info('Creating {} for {} joints', type, num_angles)
    if type == 'aa':
        decoder = AADecoder(num_angles=num_angles, **kwargs)
        dim = decoder.get_dim_size()
        ind_dim = 3
        mean = decoder.get_mean()
    elif type == 'pca':
        decoder = PCADecoder(num_pca_comps=num_pca_comps, **kwargs)
        ind_dim = num_pca_comps
        dim = decoder.get_dim_size()
        mean = decoder.get_mean()
    elif type == 'cont_rot_repr' or type == 'cont-rot-repr':
        decoder = ContinuousRotReprDecoder(num_angles, **kwargs)
        dim = decoder.get_dim_size()
        ind_dim = 6
        mean = decoder.get_mean()
    elif type == 'rot_mats':
        decoder = SVDRotationProjection()
        dim = decoder.get_dim_size()
        mean = decoder.get_mean()
        ind_dim = 9
    else:
        raise ValueError(f'Unknown pose parameterization: {type}')

    return PoseParameterization(
        decoder=decoder, dim=dim, ind_dim=ind_dim, mean=mean)
