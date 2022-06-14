import numpy as np
import torch

from .abstract_structure import AbstractStructure
from human_shape.utils.rotation_utils import batch_rodrigues

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

sign_flip = np.array(
    [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
        -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
        -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
        1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
        -1, 1, -1, -1])

SIGN_FLIP = torch.tensor([6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17,
                          12, 13, 14, 18, 19, 20, 24, 25, 26, 21, 22, 23, 27,
                          28, 29, 33, 34, 35, 30, 31, 32,
                          36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51,
                          52, 53, 48, 49, 50, 57, 58, 59, 54, 55, 56, 63, 64,
                          65, 60, 61, 62, 69, 70, 71, 66, 67, 68],
                         dtype=torch.long) - 3
SIGN_FLIP = SIGN_FLIP.detach().numpy()


class BodyPose(AbstractStructure):
    """ Stores the body pose vector. Assumes the input is in axis-angle format
    """

    def __init__(self, body_pose, **kwargs):
        super(BodyPose, self).__init__()
        self.body_pose = body_pose

    def to_tensor(self, to_rot=True, *args, **kwargs):
        self.body_pose = torch.from_numpy(self.body_pose)

        if to_rot:
            self.body_pose = batch_rodrigues(
                self.body_pose.view(-1, 3)).view(-1, 3, 3)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if torch.is_tensor(self.body_pose):
            dim_flip = torch.tensor([1, -1, -1], dtype=self.body_pose.dtype)
        else:
            dim_flip = np.array([1, -1, -1], dtype=self.body_pose.dtype)

        sign_flip = SIGN_FLIP[:self.body_pose.size]
        body_pose = (
            self.body_pose.reshape(-1)[sign_flip].reshape(-1, 3) *
            dim_flip).reshape(-1 * 3).copy()
        field = type(self)(body_pose=body_pose)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def crop(self, rot=0, *args, **kwargs):
        field = type(self)(body_pose=self.body_pose)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(rot=rot, *args, **kwargs)
            field.add_field(k, v)
        self.add_field('rot', rot)
        return field

    def to(self, *args, **kwargs):
        field = type(self)(body_pose=self.body_pose.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
