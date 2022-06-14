import numpy as np

import torch

from .abstract_structure import AbstractStructure
from human_shape.utils.rotation_utils import batch_rodrigues

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class HandPose(AbstractStructure):
    """ Stores left and right hand pose parameters
    """

    def __init__(self, left_hand_pose, right_hand_pose, **kwargs):
        super(HandPose, self).__init__()
        self.left_hand_pose = left_hand_pose
        self.right_hand_pose = right_hand_pose

    def to_tensor(self, to_rot=True, *args, **kwargs):
        if not torch.is_tensor(self.left_hand_pose):
            if self.left_hand_pose is not None:
                self.left_hand_pose = torch.from_numpy(self.left_hand_pose)
        if not torch.is_tensor(self.right_hand_pose):
            if self.right_hand_pose is not None:
                self.right_hand_pose = torch.from_numpy(
                    self.right_hand_pose)
        if to_rot:
            if self.left_hand_pose is not None:
                self.left_hand_pose = batch_rodrigues(
                    self.left_hand_pose.view(-1, 3)).view(-1, 3, 3)
            if self.right_hand_pose is not None:
                self.right_hand_pose = batch_rodrigues(
                    self.right_hand_pose.view(-1, 3)).view(-1, 3, 3)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):

        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if torch.is_tensor(self.left_hand_pose):
            dim_flip = torch.tensor([1, -1, -1], dtype=torch.float32)
        else:
            dim_flip = np.array([1, -1, -1], dtype=np.float32)

        left_hand_pose, right_hand_pose = None, None
        if self.right_hand_pose is not None:
            left_hand_pose = (self.right_hand_pose.reshape(15, 3) *
                              dim_flip).reshape(45)
        if self.left_hand_pose is not None:
            right_hand_pose = (self.left_hand_pose.reshape(15, 3) *
                               dim_flip).reshape(45)

        field = type(self)(left_hand_pose=left_hand_pose,
                           right_hand_pose=right_hand_pose)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def to(self, *args, **kwargs):
        left_hand_pose = self.left_hand_pose
        right_hand_pose = self.right_hand_pose
        if left_hand_pose is not None:
            left_hand_pose = left_hand_pose.to(*args, **kwargs)
        if right_hand_pose is not None:
            right_hand_pose = right_hand_pose.to(*args, **kwargs)
        field = type(self)(
            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
