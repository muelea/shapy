from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import torch
import cv2

from torchvision.transforms import functional as F
from .abstract_structure import AbstractStructure


class Betas(AbstractStructure):
    """ Stores the shape params
    """

    def __init__(self, betas, dtype=torch.float32, **kwargs):
        super(Betas, self).__init__()

        self.betas = betas

    def to_tensor(self, *args, **kwargs):
        if not torch.is_tensor(self.betas):
            self.betas = torch.from_numpy(self.betas)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def to(self, *args, **kwargs):
        field = type(self)(betas=self.betas.to(*args, **kwargs))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            field.add_field(k, v)
        return field
