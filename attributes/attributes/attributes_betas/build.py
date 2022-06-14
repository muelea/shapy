from typing import Union
from omegaconf import DictConfig
from sklearn.linear_model import (LinearRegression, Lasso)
from sklearn.kernel_ridge import KernelRidge
from .a2b import A2B, A2BProbabilistic
from .b2a import B2A
from .polynomial import Polynomial
from .ridge import Ridge


MODEL_DICT = {'a2b': A2B, 'b2a': B2A, 'a2b-prob': A2BProbabilistic}


def build(
    cfg: DictConfig,
    type: str = 'a2b',
) -> Union[A2B, B2A]:
    if type == 'a2b':
        return A2B(cfg)
    elif type == 'a2b-prob':
        return A2BProbabilistic(cfg)
    elif type == 'b2a':
        return B2A(cfg)
    else:
        raise ValueError(f'Unknown model: {type}')


def build_regression(
    type: str = 'linear',
    alpha: float = 0.0,
    degree: int = 3,
    kernel='linear',
    input_dim: int = 15,
    output_dim: int = 10,
):
    if type == 'linear':
        return LinearRegression()
    elif type == 'ridge':
        return Ridge(input_dim, output_dim, alpha=alpha)
    elif type == 'lasso':
        return Lasso()
    elif type == 'kernel-ridge':
        return KernelRidge(alpha=alpha,
                           kernel=kernel,
                           degree=degree)
    elif type == 'polynomial':
        return Polynomial(input_dim, output_dim, degree=degree, alpha=alpha)
    else:
        raise ValueError(f'Unknown regression type: {type}')
