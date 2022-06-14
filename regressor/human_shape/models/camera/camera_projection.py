import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from human_shape.models.body_models import KeypointTensor

DEFAULT_FOCAL_LENGTH = 5000


class CameraParams(object):
    attributes = ['translation', 'rotation', 'scale', 'focal_length',
                  'scale_first']

    KEYS = ['translation', 'rotation', 'scale', 'focal_length',
            'scale_first']

    def __init__(self, translation=None, rotation=None, scale=None,
                 scale_first=False,
                 focal_length=None):
        super(CameraParams, self).__init__()

        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.focal_length = focal_length
        self.scale_first = scale_first

    def keys(self):
        return [key for key in self.KEYS
                if getattr(self, key) is not None]

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


def build_cam_proj(camera_cfg, dtype=torch.float32):
    camera_type = camera_cfg.get('type', 'weak-persp')
    camera_pos_scale = camera_cfg.get('pos_func')
    if camera_pos_scale == 'softplus':
        camera_scale_func = F.softplus
    elif camera_pos_scale == 'exp':
        camera_scale_func = torch.exp
    elif camera_pos_scale == 'none' or camera_pos_scale == 'None':
        def func(x):
            return x
        camera_scale_func = func
    else:
        raise ValueError(
            f'Unknown positive scaling function: {camera_pos_scale}')

    if camera_type.lower() == 'persp':
        if camera_pos_scale == 'softplus':
            mean_flength = np.log(np.exp(DEFAULT_FOCAL_LENGTH) - 1)
        elif camera_pos_scale == 'exp':
            mean_flength = np.log(DEFAULT_FOCAL_LENGTH)
        elif camera_pos_scale == 'none':
            mean_flength = DEFAULT_FOCAL_LENGTH
        camera = PerspectiveCamera(dtype=dtype)
        camera_mean = torch.tensor(
            [mean_flength, 0.0, 0.0], dtype=torch.float32)
        camera_param_dim = 4
    elif camera_type.lower() == 'weak-persp':
        weak_persp_cfg = camera_cfg.get('weak_persp', {})
        scale_first = weak_persp_cfg.get('scale_first', False)
        mean_scale = weak_persp_cfg.get('mean_scale', 0.9)
        if camera_pos_scale == 'softplus':
            mean_scale = np.log(np.exp(mean_scale) - 1)
        elif camera_pos_scale == 'exp':
            mean_scale = np.log(mean_scale)
        camera_mean = torch.tensor([mean_scale, 0.0, 0.0], dtype=torch.float32)
        camera = WeakPerspectiveCamera(scale_first=scale_first, dtype=dtype)
        camera_param_dim = 3
    else:
        raise ValueError(f'Unknown camera type: {camera_type}')

    return {
        'camera': camera,
        'mean': camera_mean,
        'scale_func': camera_scale_func,
        'dim': camera_param_dim
    }


class PerspectiveCamera(nn.Module):
    ''' Module that implements a perspective camera
    '''

    FOCAL_LENGTH = DEFAULT_FOCAL_LENGTH

    def __init__(self, dtype=torch.float32, focal_length=None, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.dtype = dtype

        if focal_length is None:
            focal_length = self.FOCAL_LENGTH
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('focal_length',
                             torch.tensor(focal_length, dtype=dtype))

    def forward(self, points, focal_length=None, translation=None,
                rotation=None, camera_center=None, **kwargs):
        ''' Forward pass for the perspective camera

            Parameters
            ----------
                points: torch.tensor, BxNx3
                    The tensor that contains the points that will be projected.
                    If not in homogeneous coordinates, then
                focal_length: torch.tensor, BxNx3, optional
                    The predicted focal length of the camera. If not given,
                    then the default value of 5000 is assigned
                translation: torch.tensor, Bx3, optional
                    The translation predicted for each element in the batch. If
                    not given  then a zero translation vector is assumed
                rotation: torch.tensor, Bx3x3, optional
                    The rotation predicted for each element in the batch. If
                    not given  then an identity rotation matrix is assumed
                camera_center: torch.tensor, Bx2, optional
                    The center of each image for the projection. If not given,
                    then a zero vector is used
            Returns
            -------
                Returns a torch.tensor object with size BxNx2 with the
                location of the projected points on the image plane
        '''

        device = points.device
        batch_size = points.shape[0]

        if rotation is None:
            rotation = torch.eye(
                3, dtype=points.dtype, device=device).unsqueeze(dim=0).expand(
                    batch_size, -1, -1)
        if translation is None:
            translation = torch.zeros(
                [3], dtype=points.dtype,
                device=device).unsqueeze(dim=0).expand(batch_size, -11)

        if camera_center is None:
            camera_center = torch.zeros([batch_size, 2], dtype=points.dtype,
                                        device=device)

        with torch.no_grad():
            camera_mat = torch.zeros([batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            if focal_length is None:
                focal_length = self.focal_length

            camera_mat[:, 0, 0] = focal_length
            camera_mat[:, 1, 1] = focal_length

        points_transf = torch.einsum(
            'bji,bmi->bmj',
            rotation, points) + translation.unsqueeze(dim=1)

        img_points = torch.div(points_transf[:, :, :2],
                               points_transf[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum(
            'bmi,bji->bjm',
            camera_mat, img_points) + camera_center.reshape(-1, 1, 2)
        return img_points


class WeakPerspectiveCamera(nn.Module):
    ''' Scaled Orthographic / Weak-Perspective Camera
    '''

    def __init__(self, scale_first=False, **kwargs):
        super(WeakPerspectiveCamera, self).__init__()
        self.scale_first = scale_first

    def forward(self, points, scale, translation, **kwargs):
        ''' Implements the forward pass for a Scaled Orthographic Camera

            Parameters
            ----------
                points: torch.tensor, BxNx3
                    The tensor that contains the points that will be projected.
                    If not in homogeneous coordinates, then
                scale: torch.tensor, Bx1
                    The predicted scaling parameters
                translation: torch.tensor, Bx2
                    The translation applied on the image plane to the points
            Returns
            -------
                projected_points: torch.tensor, BxNx2
                    The points projected on the image plane, according to the
                    given scale and translation
        '''
        assert translation.shape[-1] == 2, 'Translation shape must be -1x2'
        assert scale.shape[-1] == 1, 'Scale shape must be -1x1'

        if self.scale_first:
            projected_points = (
                scale.view(-1, 1, 1) * points[:, :, :2]) + translation.view(
                    -1, 1, 2)
        else:
            projected_points = scale.view(-1, 1, 1) * (
                points[:, :, :2] + translation.view(-1, 1, 2))
        if (type(projected_points) != type(points) and isinstance(
                points, (KeypointTensor,))):
            projected_points = KeypointTensor.from_obj(
                projected_points, points)
        return projected_points
