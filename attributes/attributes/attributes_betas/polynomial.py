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


class Polynomial(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int = 2,
        alpha: float = 0.0,
    ) -> None:
        super(Polynomial, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = alpha

        x = torch.rand([input_dim])
        y = torch.vander(x, degree)

        combinations = list(
            self._combinations(input_dim, degree, False, False))

        num_input = len(combinations)
        self.coeff_size = len(combinations)
        self.linear = nn.Linear(num_input, output_dim)
        logger.info(self.linear)
        for ii in range(degree):
            indices = []
            for c in combinations:
                if len(c) == ii + 1:
                    indices.append(c)
            indices = torch.tensor(indices, dtype=torch.long)
            # logger.info(f'{ii + 1} : {indices}')
            self.register_buffer(f'indices_{ii:03d}', indices)

    @staticmethod
    def _combinations(n_features, degree, interaction_only, include_bias):
        start = int(not include_bias)
        return chain.from_iterable(combinations_w_r(range(n_features), i)
                                   for i in range(start, degree + 1))

    def build_polynomial_coeffs(self, X, degree=2):
        A = []
        for ii in range(self.degree):
            indices = getattr(self, f'indices_{ii:03d}')
            values = X[:, indices]
            A.append(torch.prod(values, dim=-1))

        A = torch.cat(A, dim=-1)
        return A

    def fit(self, X, Y):
        self.sk_polynomial = make_pipeline(
            PolynomialFeatures(degree=self.degree),
            Ridge(alpha=self.alpha, fit_intercept=False)
        )
        self.sk_polynomial.fit(X, Y)

        params = self.sk_polynomial.get_params()
        for key, val in params.items():
            if not (key == 'ridge'):
                continue

            with torch.no_grad():
                self.linear.weight.data[:] = torch.from_numpy(
                    val.coef_[:, 1:])
                self.linear.bias.data[:] = torch.from_numpy(
                    val.coef_[:, 0])
        return self

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Union[str, IO],
        map_location: Optional[
            Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        strict: bool = True
    ):
        ckpt_dict = torch.load(checkpoint_path)
        hparams = ckpt_dict['hparams']

        obj = Polynomial(**hparams)
        logger.info(obj)

        obj.load_state_dict(ckpt_dict['model'])

        return obj

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, IO],
        map_location: Optional[
            Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        strict: bool = True
    ):
        ckpt_dict = {'model': self.state_dict(),
                     'hparams': {
                     'input_dim': self.input_dim,
                     'output_dim': self.output_dim,
                     'alpha': self.alpha,
                     'degree': self.degree,
        }}
        torch.save(ckpt_dict, checkpoint_path)

    def predict(self, X: Union[Array, Tensor]):
        to_numpy = False
        if not torch.is_tensor(X):
            to_numpy = True
            X = torch.from_numpy(X).to(dtype=torch.float32)
        output = self(X)
        # output = self.sk_polynomial.predict(X)
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
