import numpy as np
import torch

import cv2
from loguru import logger
from .abstract_structure import AbstractStructure

from human_shape.utils import Array, Tensor, IntTuple, get_transform

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Points2D(AbstractStructure):
    """ Stores a 2D point grid
    """

    def __init__(
        self,
        points,
        size: IntTuple,
        flip_axis=0,
        dtype=torch.float32,
        bc=None,
        closest_faces=None,
    ) -> None:
        super(Points2D, self).__init__()
        self.points = points
        self.size = size
        self.flip_axis = flip_axis
        self.closest_faces = closest_faces
        self.bc = bc

    def __getitem__(self, key):
        if key == 'points':
            return self.points
        else:
            raise ValueError(f'Unknown key: {key}')

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        width = self.size[1]
        TO_REMOVE = 1
        flipped_points = self.points.copy()
        flipped_points[:, self.flip_axis] = (
            width - flipped_points[:, self.flip_axis] - TO_REMOVE)

        if self.bc is not None:
            closest_tri_points = flipped_points[self.closest_faces].copy()
            flipped_points = (
                self.bc[:, :, np.newaxis] * closest_tri_points).sum(axis=1)
            flipped_points = flipped_points.astype(self.points.dtype)

        points = type(self)(flipped_points,
                            size=self.size,
                            flip_axis=self.flip_axis,
                            bc=self.bc,
                            closest_faces=self.closest_faces,
                            )

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            points.add_field(k, v)
        self.add_field('is_flipped', True)
        return points

    def to_tensor(self, *args, **kwargs):
        self.points = torch.from_numpy(self.points)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def shift(self, vector, *args, **kwargs):
        points = self.points.copy()
        points += vector.reshape(1, 2)

        field = type(self)(points,
                           self.size,
                           flip_axis=self.flip_axis,
                           bc=self.bc,
                           closest_faces=self.closest_faces)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.shift(vector, *args, **kwargs)
            field.add_field(k, v)
        return field

    def crop(self, center, scale, crop_size=224, *args, **kwargs):
        points = self.points.copy()
        transf = get_transform(center, scale, (crop_size, crop_size))
        points = (np.dot(
            points, transf[:2, :2].T) + transf[:2, 2] + 1).astype(points.dtype)

        field = type(self)(points,
                           (crop_size, crop_size, 3),
                           flip_axis=self.flip_axis,
                           bc=self.bc,
                           closest_faces=self.closest_faces)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(*args, **kwargs)
            field.add_field(k, v)

        self.add_field('rot', kwargs.get('rot', 0))
        return field

    def rotate(self, rot=0, *args, **kwargs):
        if rot == 0:
            return self
        points = self.points.copy()
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
        points = (np.dot(points, M[:2, :2].T) + M[:2, 2] + 1).astype(
            points.dtype)

        points = type(self)(
            points, size=self.size, flip_axis=self.flip_axis,
            bc=self.bc,
            closest_faces=self.closest_faces,)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.rotate(rot=rot, *args, **kwargs)
            points.add_field(k, v)

        self.add_field('rot', rot)
        return points

    def as_array(self) -> Array:
        if torch.is_tensor(self.points):
            points = self.points.detach().cpu().numpy()
        else:
            points = self.points.copy()
        return points

    def as_tensor(self, dtype=torch.float32, device=None) -> Tensor:
        if torch.is_tensor(self.points):
            return self.points
        else:
            return torch.tensor(self.points, dtype=dtype, device=device)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig)
                       for s, s_orig in zip(size, self.size))
        ratio_h, ratio_w, _ = ratios
        resized_data = self.points.copy()

        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h

        points = type(self)(resized_data,
                            size=size,
                            flip_axis=self.flip_axis,
                            bc=self.bc,
                            closest_faces=self.closest_faces,)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.resize(size, *args, **kwargs)
            points.add_field(k, v)

        return points

    def to(self, *args, **kwargs):
        points = type(self)(
            self.points.to(*args, **kwargs),
            size=self.size, flip_axis=self.flip_axis)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            points.add_field(k, v)
        return points

    def normalize(self, *args, **kwargs):
        if torch.is_tensor(self.points):
            points = self.points.clone()
        else:
            points = self.points.copy()

        H, W, _ = self.size
        points[:, 0] = 2.0 * points[:, 0] / W - 1.0
        points[:, 1] = 2.0 * points[:, 1] / H - 1.0

        points = type(self)(points, size=self.size, flip_axis=self.flip_axis,
                            bc=self.bc,
                            closest_faces=self.closest_faces,)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.normalize(*args, **kwargs)
            points.add_field(k, v)

        return points
