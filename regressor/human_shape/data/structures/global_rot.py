import numpy as np

import torch
import cv2
from .abstract_structure import AbstractStructure
from human_shape.utils.rotation_utils import batch_rodrigues

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class GlobalRot(AbstractStructure):

    def __init__(self, global_rot, **kwargs):
        super(GlobalRot, self).__init__()
        self.global_rot = global_rot

    def to_tensor(self, to_rot=True, *args, **kwargs):
        if not torch.is_tensor(self.global_rot):
            self.global_rot = torch.from_numpy(self.global_rot)

        if to_rot:
            self.global_rot = batch_rodrigues(
                self.global_rot.view(-1, 3)).view(1, 3, 3)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):

        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if torch.is_tensor(self.global_rot):
            dim_flip = torch.tensor([1, -1, -1], dtype=self.global_rot.dtype)
            global_rot = self.global_rot.clone().squeeze() * dim_flip
        else:
            dim_flip = np.array([1, -1, -1], dtype=self.global_rot.dtype)
            global_rot = self.global_rot.copy().squeeze() * dim_flip

        field = type(self)(global_rot=global_rot)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def rotate(self, rot=0, *args, **kwargs):
        global_rot = self.global_rot.copy()
        if rot != 0:
            R = np.array([[np.cos(np.deg2rad(-rot)),
                           -np.sin(np.deg2rad(-rot)), 0],
                          [np.sin(np.deg2rad(-rot)),
                           np.cos(np.deg2rad(-rot)), 0],
                          [0, 0, 1]], dtype=np.float32)

            # find the rotation of the body in camera frame
            per_rdg, _ = cv2.Rodrigues(global_rot)
            # apply the global rotation to the global orientation
            resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
            global_rot = (resrot.T)[0].reshape(3)
        field = type(self)(global_rot=global_rot)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(rot=rot, *args, **kwargs)
            field.add_field(k, v)

        self.add_field('rot', rot)
        return field

    def to(self, *args, **kwargs):
        field = type(self)(global_rot=self.global_rot.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
