from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import time
from typing import Callable, Iterator, Union, Optional, List

import os.path as osp
import yaml
from loguru import logger

import pickle

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


def build_robustifier(robustifier_type: str = None, **kwargs) -> nn.Module:
    if robustifier_type is None or robustifier_type == 'none':
        return None
    elif robustifier_type == 'gmof':
        return GMOF(**kwargs)
    elif robustifier_type == 'charbonnier':
        return Charbonnier(**kwargs)
    elif robustifier_type == 'wing':
        return Wing(**kwargs)
    else:
        raise ValueError(f'Unknown robustifier: {robustifier_type}')


class GMOF(nn.Module):
    def __init__(self, rho: float = 100, **kwargs) -> None:
        super(GMOF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return f'Rho = {self.rho}'

    def forward(self, residual):
        squared_residual = residual.pow(2)
        return torch.div(squared_residual, squared_residual + self.rho ** 2)


class Charbonnier(nn.Module):
    def __init__(self, epsilon: float = 1e-6, **kwargs) -> None:
        super(Charbonnier, self).__init__()
        self.epsilon = epsilon

    def extra_repr(self) -> str:
        return f'Epsilon = {self.epsilon}'

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return (residual.pow(2) + self.epsilon ** 2).sqrt()


class Wing(nn.Module):
    ''' Implements the Wing loss

        First used in:
        @inproceedings{feng2018wing,
          title={Wing loss for robust facial landmark
          localisation with convolutional neural networks},
          author={Feng, Zhen-Hua and Kittler, Josef and
          Awais, Muhammad and Huber, Patrik and Wu, Xiao-Jun},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and
          Pattern Recognition},
          pages={2235--2245},
          year={2018}
        }
    '''

    def __init__(
        self,
        threshold: float = 0.1,
        scale: float = 1.0,
        epsilon: float = 2.0,
        **kwargs
    ) -> None:
        super(Wing, self).__init__()
        self.scale = scale
        self.threshold = threshold
        self.constant = self.threshold - threshold * np.log(
            1 + self.threshold / self.scale)

    def extra_repr(self) -> str:
        msg = []
        msg.append(f'Threshold: {self.threshold}')
        msg.append(f'Constant: {self.constant}')
        msg.append(f'Scale: {self.scale}')
        return '\n'.join(msg)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        abs_residual = residual.abs()
        cond = abs_residual < self.threshold

        loss = torch.where(
            cond,
            self.threshold * torch.log(1 + abs_residual / self.scale),
            abs_residual - self.constant
        )
        return loss
