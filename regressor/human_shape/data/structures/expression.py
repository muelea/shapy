import numpy as np
from copy import deepcopy

import torch

from .abstract_structure import AbstractStructure


class Expression(AbstractStructure):
    """ Stores the expression params
    """

    def __init__(self, expression, dtype=torch.float32, **kwargs):
        super(Expression, self).__init__()
        self.expression = expression

    def to_tensor(self, *args, **kwargs):
        if not torch.is_tensor(self.expression):
            self.expression = torch.from_numpy(self.expression)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def transpose(self, method):
        field = type(self)(expression=deepcopy(self.expression))
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            field.add_field(k, v)
        self.add_field('is_flipped', True)
        return field

    def resize(self, size, *args, **kwargs):
        field = type(self)(expression=self.expression)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.resize(size, *args, **kwargs)
            field.add_field(k, v)
        return field

    def crop(self, rot=0, *args, **kwargs):
        field = type(self)(expression=self.expression)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(rot=rot, *args, **kwargs)
            field.add_field(k, v)

        self.add_field('rot', rot)
        return field

    def to(self, *args, **kwargs):
        field = type(self)(expression=self.expression.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
