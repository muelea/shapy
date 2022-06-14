from typing import Optional, Union, Tuple

import numpy as np

import cv2

import torch
from loguru import logger

from .abstract_structure import AbstractStructure
from ..utils import (
    map_keypoints,
    keyps_to_bbox, bbox_to_center_scale,
    KEYPOINT_CONNECTIONS_DICT, KEYPOINT_NAMES_DICT,
    KEYPOINT_PARTS_DICT,
)

from human_shape.utils.transf_utils import get_transform
from human_shape.utils import Array, Tensor, StringList, IntTuple

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Keypoints2D(AbstractStructure):
    def __init__(
        self,
        keypoints: Union[Tensor, Array],
        size: IntTuple,
        flip_indices: Array,
        flip_axis: int = 0,
        source: str = '',
        apply_crop: bool = True,
        **kwargs
    ) -> None:
        super(Keypoints2D, self).__init__()
        self.size = size
        self.source = source
        self.keypoints = keypoints[:, :-1]
        self.conf = keypoints[:, -1]
        self.flip_indices = flip_indices
        self.flip_axis = flip_axis
        self.apply_crop = apply_crop
        # Keypoint dimension
        self.dim = 2

        self._names = KEYPOINT_NAMES_DICT.get(self.source, None)
        if self._names is None:
            error_msg = (f'{self.source} not in names dict.'
                         f' Please make sure that {self.source} is a key'
                         f' in the keypoint dictionary!')
            raise ValueError(error_msg)
        _connections = KEYPOINT_CONNECTIONS_DICT.get(self.source, None)
        if _connections is None:
            error_msg = (f'{self.source} not in connections dict.'
                         f' Please make sure that {self.source} is a key'
                         f' in the keypoint dictionary!')
            raise ValueError(error_msg)
        self._connections = _connections

        _parts = KEYPOINT_PARTS_DICT.get(self.source, None)
        if _parts is None:
            error_msg = (f'{self.source} not in parts dict.'
                         f' Please make sure that {self.source} is a key'
                         f' in the keypoint dictionary!')
            raise ValueError(error_msg)
        self._parts = _parts

    @property
    def parts(self):
        return self._parts

    @property
    def connections(self):
        return self._connections

    @property
    def names(self):
        return self._names

    def as_tensor(self) -> Tensor:
        pass

    def as_array(self, scale=False) -> Array:
        keypoints = self.keypoints
        conf = self.conf
        if torch.is_tensor(keypoints):
            keypoints = keypoints.detach().cpu().numpy()
            conf = conf.detach().cpu().numpy()
        if scale:
            H, W, _ = self.size
            keypoints[:, 0] = (keypoints[:, 0] + 1) * 0.5 * W
            keypoints[:, 1] = (keypoints[:, 1] + 1) * 0.5 * H

        return np.concatenate([keypoints, conf[:, None]], axis=1)

    def torso_crop(self):
        dset_scale_factor = self.get_field('dset_scale_factor')
        torso_idxs = self.parts['torso']
        indexes = torso_idxs[torso_idxs < len(self.conf)]
        torso_body_keyps = self.keypoints[indexes]
        torso_body_conf = self.conf[indexes]
        bbox = keyps_to_bbox(
            torso_body_keyps, torso_body_conf, img_size=self.size)
        center, scale, bbox_size = bbox_to_center_scale(
            bbox, dset_scale_factor=dset_scale_factor,
        )
        return center, scale, bbox_size

    def upper_body_crop(self, dset_scale_factor=1.0):
        upper_body_idxs = self.parts['upper']
        indexes = upper_body_idxs[upper_body_idxs < len(self.conf)]
        upper_keyps = self.keypoints[indexes]
        upper_conf = self.conf[indexes]
        bbox = keyps_to_bbox(
            upper_keyps, upper_conf, img_size=self.size)
        center, scale, bbox_size = bbox_to_center_scale(
            bbox, dset_scale_factor=dset_scale_factor,
        )
        return center, scale, bbox_size

    def to_dset(
        self,
        target_dataset: str,
        output: str = 'tensor',
        source_names: Optional[StringList] = None,
        target_names: Optional[StringList] = None,
    ) -> Union[Tensor, Array, AbstractStructure]:
        ''' Maps the keypoints from the source to a target dataset format
        '''
        target_indices, source_indices, target_dim = map_keypoints(
            self.source,
            target_dataset,
            KEYPOINT_NAMES_DICT,
            target_names=target_names,
            source_names=source_names)

        VALID_OUTPUTS = ['tensor', 'array', 'structure']
        assert output in VALID_OUTPUTS, (
            f'Output must me one of {VALID_OUTPUTS}, got {output}'
        )

        if torch.is_tensor(self.keypoints):
            dtype, device = self.keypoints.dtype, self.keypoints.device
            output_keypoints = torch.zeros(
                [target_dim, self.dim], dtype=dtype, device=device)
            output_conf = torch.zeros(
                [target_dim], dtype=dtype, device=device)
            output_keypoints[target_indices] = self.keypoints[source_indices]
            output_conf[target_indices] = self.conf[source_indices]
            output_conf = output_conf.view(-1, 1)
        else:
            dtype = self.keypoints.dtype.keypoints.device
            output_keypoints = np.zeros(
                [target_dim, self.dim], dtype=dtype)
            output_conf = torch.zeros([target_dim, 1], dtype=dtype)

            output_keypoints[target_indices] = self.keypoints[source_indices]
            output_conf[target_indices] = self.conf[source_indices]

        if output == 'tensor':
            if not torch.is_tensor(output_keypoints):
                output_keypoints = torch.from_numpy(
                    output_keypoints).to(dtype=torch.float32)
                output_conf = torch.from_numpy(
                    output_conf).to(dtype=torch.float32)
            return torch.cat([output_keypoints, output_conf], dim=-1)
        elif output == 'array':
            if torch.is_tensor(output_keypoints):
                output_keypoints = output_keypoints.detach().cpu().numpy()
                output_conf = output_conf.detach().cpu().numpy()
            return np.concatenate([output_keypoints, output_conf], axis=-1)
        else:
            if torch.is_tensor(output_keypoints):
                keyps = torch.cat([output_keypoints, output_conf], dim=-1)
            else:
                keyps = np.concatenate(
                    [output_keypoints, output_conf], axis=-1)
            return Keypoints2D(keyps, size=self.size,
                               flip_indices=self.flip_indices,
                               flip_axis=self.flip_axis,
                               source=target_dataset,
                               apply_crop=self.apply_crop,
                               )

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'Number of keypoints={}, '.format(self.keypoints.shape[0])
        s += 'image_width={}, '.format(self.size[1])
        s += 'image_height={})'.format(self.size[0])
        return s

    def to_tensor(self, *args, **kwargs):
        if not torch.is_tensor(self.keypoints):
            self.keypoints = torch.from_numpy(self.keypoints)
            self.conf = torch.from_numpy(self.conf)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def rotate(self, rot=0, *args, **kwargs):
        (h, w) = self.size[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        kp = self.keypoints.copy()
        kp = (np.dot(kp, M[:2, :2].T) + M[:2, 2] + 1).astype(np.int)

        conf = self.conf.copy().reshape(-1, 1)
        kp = np.concatenate([kp, conf], axis=1).astype(np.float32)
        keypoints = type(self)(kp, size=(nH, nW, 3),
                               flip_indices=self.flip_indices,
                               source=self.source,
                               apply_crop=self.apply_crop,
                               flip_axis=self.flip_axis)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.rotate(rot=rot, *args, **kwargs)
            keypoints.add_field(k, v)

        self.add_field('rot', rot)
        return keypoints

    def shift(self, vector, *args, **kwargs):
        # Shift the keypoints using the motion vector
        if torch.is_tensor(self.keypoints):
            kp = self.keypoints.clone()
            conf = self.conf.clone().reshape(-1, 1)
        else:
            kp = self.keypoints.copy()
            conf = self.conf.copy().reshape(-1, 1)
        kp = kp + vector.reshape(1, 2)
        kp = np.concatenate([kp, conf], axis=1).astype(np.float32)

        keypoints = type(self)(kp, size=self.size,
                               flip_indices=self.flip_indices,
                               source=self.source,
                               apply_crop=self.apply_crop,
                               flip_axis=self.flip_axis)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.shift(vector)
            keypoints.add_field(k, v)
        keypoints.add_field('motion_blur_shift', vector)
        return keypoints

    def crop(self, center, scale, crop_size=224, *args, **kwargs):
        if not self.apply_crop:
            for k, v in self.extra_fields.items():
                if isinstance(v, AbstractStructure):
                    v = v.crop(center=center, scale=scale,
                               crop_size=crop_size, *args, **kwargs)
                self.add_field(k, v)
            return self
        kp = self.keypoints.copy()
        transf = get_transform(center, scale, (crop_size, crop_size))
        kp = (np.dot(kp, transf[:2, :2].T) + transf[:2, 2] + 1).astype(np.int)
        conf = self.conf.copy().reshape(-1, 1)

        kp = np.concatenate([kp, conf], axis=1).astype(np.float32)
        keypoints = type(self)(kp, size=(crop_size, crop_size, 3),
                               flip_indices=self.flip_indices,
                               source=self.source,
                               apply_crop=self.apply_crop,
                               flip_axis=self.flip_axis)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(center=center, scale=scale,
                           crop_size=crop_size, *args, **kwargs)
            keypoints.add_field(k, v)

        return keypoints

    def normalize(self, *args, **kwargs):
        if torch.is_tensor(self.keypoints):
            kp = self.keypoints.clone()
            conf = self.conf.clone().reshape(-1, 1)
        else:
            kp = self.keypoints.copy()
            conf = self.conf.copy().reshape(-1, 1)

        H, W, _ = self.size
        kp[:, 0] = 2.0 * kp[:, 0] / W - 1.0
        kp[:, 1] = 2.0 * kp[:, 1] / H - 1.0
        if torch.is_tensor(self.keypoints):
            kp = torch.cat([kp, conf], dim=1)
        else:
            kp = np.concatenate([kp, conf], axis=1).astype(np.float32)

        keypoints = type(self)(kp, size=self.size,
                               flip_indices=self.flip_indices,
                               source=self.source,
                               apply_crop=self.apply_crop,
                               flip_axis=self.flip_axis)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.normalize(*args, **kwargs)
            keypoints.add_field(k, v)

        return keypoints

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig)
                       for s, s_orig in zip(size, self.size))
        ratio_h, ratio_w, _ = ratios
        resized_data = self.keypoints.copy()

        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h

        resized_keyps = np.concatenate(
            [resized_data, self.conf[:, np.newaxis]], axis=-1)

        keypoints = type(self)(resized_keyps,
                               source=self.source,
                               size=size,
                               apply_crop=self.apply_crop,
                               flip_indices=self.flip_indices,
                               flip_axis=self.flip_axis)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.resize(size, *args, **kwargs)
            keypoints.add_field(k, v)

        return keypoints

    def __getitem__(self, key):
        if key == 'keypoints':
            return self.keypoints
        elif key == 'conf':
            return self.conf
        else:
            raise ValueError('Unknown key: {}'.format(key))

    def __len__(self):
        return 1

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT implemented")

        width = self.size[1]
        TO_REMOVE = 1
        if torch.is_tensor(self.keypoints):
            flipped_data = torch.cat([self.keypoints,
                                      self.conf.unsqueeze(dim=-1)],
                                     dim=-1)

            num_joints = flipped_data.shape[0]
            flipped_data[np.arange(num_joints)] = flipped_data[
                self.flip_indices[:num_joints]]
            #  TO_REMOVE = 1
            flipped_data[..., :, self.flip_axis] = width - flipped_data[
                ..., :, self.flip_axis] - TO_REMOVE
        else:
            flipped_data = np.concatenate(
                [self.keypoints, self.conf[..., np.newaxis]], axis=-1)

            num_joints = flipped_data.shape[0]
            flipped_data[np.arange(num_joints)] = flipped_data[
                self.flip_indices[:num_joints]]
            # Flip x coordinates
            flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        keypoints = type(self)(flipped_data, self.size,
                               source=self.source,
                               flip_indices=self.flip_indices,
                               apply_crop=self.apply_crop,
                               flip_axis=self.flip_axis)
        keypoints.source = self.source

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            keypoints.add_field(k, v)

        self.add_field('is_flipped', True)
        return keypoints

    def to(self, *args, **kwargs):
        keyp_tensor = torch.cat([self.keypoints,
                                 self.conf.unsqueeze(dim=-1)], dim=-1)
        keypoints = type(self)(keyp_tensor.to(*args, **kwargs), self.size,
                               source=self.source,
                               flip_indices=self.flip_indices,
                               apply_crop=self.apply_crop,
                               flip_axis=self.flip_axis,
                               )
        keypoints.source = self.source
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints


class Keypoints3D(Keypoints2D):
    def __init__(self, *args, **kwargs):
        super(Keypoints3D, self).__init__(*args, **kwargs)
        # Keypoint dimension
        self.dim = 3

    def shift(self, vector, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.shift(vector)
            self.add_field(k, v)
        self.add_field('motion_blur_shift', vector)
        return self

    def normalize(self, *args, **kwargs):
        keypoints = self
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.normalize(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def rotate(self, rot=0, *args, **kwargs):
        kp = self.keypoints.copy()
        conf = self.conf.copy().reshape(-1, 1)

        if rot != 0:
            R = np.array([[np.cos(np.deg2rad(-rot)),
                           -np.sin(np.deg2rad(-rot)), 0],
                          [np.sin(np.deg2rad(-rot)),
                           np.cos(np.deg2rad(-rot)), 0],
                          [0, 0, 1]], dtype=np.float32)
            kp = np.dot(kp, R.T)

        kp = np.concatenate([kp, conf], axis=1).astype(np.float32)

        keypoints = type(self)(kp, size=self.size,
                               source=self.source,
                               flip_indices=self.flip_indices,
                               flip_axis=self.flip_axis)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.rotate(rot=rot, *args, **kwargs)
            keypoints.add_field(k, v)
        self.add_field('rot', kwargs.get('rot', 0))
        return keypoints

    def resize(self, size, *args, **kwargs):
        resized_keyps = np.concatenate(
            [self.keypoints, self.conf[:, np.newaxis]], axis=-1)

        keypoints = type(self)(resized_keyps,
                               source=self.source,
                               size=size,
                               flip_indices=self.flip_indices,
                               flip_axis=self.flip_axis)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.resize(size, *args, **kwargs)
            keypoints.add_field(k, v)

        return keypoints

    def crop(self, center, scale, crop_size=224, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(center=center, scale=scale,
                           crop_size=crop_size, *args, **kwargs)
        return self

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT implemented")

        if torch.is_tensor(self.keypoints):
            flipped_data = torch.cat([self.keypoints,
                                      self.conf.unsqueeze(dim=-1)],
                                     dim=-1)

            num_joints = flipped_data.shape[0]
            #  flipped_data[torch.arange(num_joints)] = torch.index_select(
            #  flipped_data, 0, flip_inds[:num_joints])
            flipped_data[np.arange(num_joints)] = flipped_data[
                self.flip_indices[:num_joints]]
            #  width = self.size[0]
            #  TO_REMOVE = 1
            # Flip x coordinates
            #  flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE
            flipped_data[..., :, self.flip_axis] *= (-1)

            #  Maintain COCO convention that if visibility == 0, then x, y = 0
            #  inds = flipped_data[..., 2] == 0
            #  flipped_data[inds] = 0
        else:
            flipped_data = np.concatenate(
                [self.keypoints, self.conf[..., np.newaxis]], axis=-1)

            num_joints = flipped_data.shape[0]
            #  flipped_data[torch.arange(num_joints)] = torch.index_select(
            #  flipped_data, 0, flip_inds[:num_joints])
            flipped_data[np.arange(num_joints)] = flipped_data[
                self.flip_indices[:num_joints]]
            #  width = self.size[0]
            #  TO_REMOVE = 1
            # Flip x coordinates
            #  flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE
            flipped_data[..., :, self.flip_axis] *= (-1)

        keypoints = type(self)(flipped_data, self.size,
                               source=self.source,
                               flip_indices=self.flip_indices,
                               flip_axis=self.flip_axis)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            keypoints.add_field(k, v)
        self.add_field('is_flipped', True)

        return keypoints
