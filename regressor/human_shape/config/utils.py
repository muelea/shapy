from typing import Tuple
from copy import deepcopy
from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class FScores:
    hand: Tuple[float] = (5.0 / 1000, 15.0 / 1000)
    head: Tuple[float] = (5.0 / 1000, 15.0 / 1000)


@dataclass
class Variable:
    create: bool = True
    requires_grad: bool = True


@dataclass
class Pose(Variable):
    type: str = 'cont-rot-repr'


@dataclass
class Normalization:
    type: str = 'batch-norm'
    affine: bool = True
    elementwise_affine: bool = True


@dataclass
class LeakyRelu:
    negative_slope: float = 0.01


@dataclass
class Activation:
    type: str = 'relu'
    leaky_relu: LeakyRelu = LeakyRelu()
