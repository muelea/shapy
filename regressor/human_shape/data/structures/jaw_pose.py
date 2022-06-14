import numpy as np

import torch

from loguru import logger
from .abstract_structure import AbstractStructure
from human_shape.utils.rotation_utils import batch_rodrigues

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class JawPose(AbstractStructure):
    """ Stores the jaw pose parameters. Assumes the input is in axis-angle.
    """

    def __init__(self, jaw_pose, dtype=torch.float32, **kwargs):
        super(JawPose, self).__init__()
        self.jaw_pose = jaw_pose

    def to_tensor(self, to_rot=True, *args, **kwargs):
        if not torch.is_tensor(self.jaw_pose):
            self.jaw_pose = torch.from_numpy(self.jaw_pose)

        if to_rot:
            self.jaw_pose = batch_rodrigues(
                self.jaw_pose.view(-1, 3)).view(-1, 3, 3)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):

        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        dim_flip = np.array([1, -1, -1], dtype=self.jaw_pose.dtype)
        jaw_pose = self.jaw_pose.copy() * dim_flip

        field = type(self)(jaw_pose=jaw_pose)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def to(self, *args, **kwargs):
        field = type(self)(jaw_pose=self.jaw_pose.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
