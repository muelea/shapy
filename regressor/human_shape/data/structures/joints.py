import numpy as np

import torch
from .abstract_structure import AbstractStructure


class Joints(AbstractStructure):
    def __init__(self, joints, **kwargs):
        super(Joints, self).__init__()
        #  self.joints = to_tensor(joints)
        self.joints = joints

    def __repr__(self):
        s = self.__class__.__name__
        return s

    def to_tensor(self, *args, **kwargs):
        self.joints = torch.tensor(self.joints)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def __getitem__(self, key):
        if key == 'joints':
            return self.joints
        else:
            raise ValueError('Unknown key: {}'.format(key))

    def __len__(self):
        return 1

    def to(self, *args, **kwargs):
        joints = type(self)(self.joints.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            joints.add_field(k, v)
        return joints
