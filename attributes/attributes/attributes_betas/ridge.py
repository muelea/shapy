from typing import Tuple, Optional, Union, Dict, Callable, IO
import sys
import os
import os.path as osp

from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

from attributes.utils.typing import Tensor, Array


class Ridge(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alpha: float = 0.0,
    ) -> None:
        super(Ridge, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.linear = nn.Linear(input_dim, output_dim)

    def fit(self, X, Y):
        logger.info(f'{self.degree}, {self.alpha}')
        self.sk_model = Ridge(alpha=self.alpha)
        self.sk_model.fit(X, Y)

        params = self.sk_model.get_params()
        for key, val in params.items():
            if not (key == 'ridge'):
                continue

            with torch.no_grad():
                self.linear.weight.data[:] = torch.from_numpy(
                    val.coef_[:, 1:])
                self.linear.bias.data[:] = torch.from_numpy(
                    val.coef_[:, 0])
        return self

    def predict(self, X: Union[Array, Tensor]):
        to_numpy = False
        if not torch.is_tensor(X):
            to_numpy = True
            X = torch.from_numpy(X).to(dtype=torch.float32)
        output = self(X)
        # output = self.sk_model.predict(X)
        if to_numpy:
            # return output
            return output.detach().cpu().numpy()
        else:
            # return torch.from_numpy(output).to(dtype=torch.float32)
            return output

    def forward(self, x):
        A = self.build_polynomial_coeffs(x, degree=self.degree)
        output = self.linear(A)
        return output
