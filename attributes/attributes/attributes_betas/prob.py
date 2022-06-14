from typing import Optional, Tuple, List, Union, Dict, Iterable
import sys

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.linalg import lu

from omegaconf import DictConfig

from loguru import logger

from nflows import transforms, distributions, flows


from .models import build_network

from attributes.utils.typing import Tensor, IntList


def build_scale_foo(scale_func: str = 'softplus'):
    if scale_func == 'softplus':
        scale_func = F.softplus

        def inv_softplus(y):
            if torch.is_tensor(y):
                x = (y.exp() - 1).log()
            else:
                x = np.log(np.exp(y) - 1)
            return x

        inv = inv_softplus
    elif scale_func == 'exp':
        return torch.exp, torch.log
    elif scale_func == 'exp':
        def squareplus(x):
            if torch.is_tensor(x):
                return 0.5 * (x + (x.pow(2) + 4).sqrt())
            else:
                return 0.5 * (x + np.sqrt(x ** 2 + 4))

        return squareplus, None

    return scale_func, inv


class MultiVariateNormalRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        distr_dim: int,
        scale_func: str = 'softplus',
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super(MultiVariateNormalRegressor, self).__init__()

        network_cfg = cfg.get('network', {})

        prob_cfg = cfg.get('probabilistic', {})
        gauss_cfg = prob_cfg.get('gaussian', {})
        covariance = gauss_cfg.get('covariance', 'diagonal')

        self.distr_dim = distr_dim
        self.covariance_type = covariance

        self.scale_func, _ = build_scale_foo(scale_func)

        if self.covariance_type == 'diagonal':
            # If we want to predict a diagonal covariance the output dimension
            # is two times the dimension, i.e. n elements for the mean and n
            # for the
            output_dim = distr_dim + distr_dim
            idxs = torch.arange(0, distr_dim, dtype=torch.long)
            self.register_buffer('row_idxs', idxs)
            self.register_buffer('col_idxs', idxs)
        elif self.covariance_type == 'tril':
            # If we want to predict a diagonal covariance the output dimension
            # is two times the dimension, i.e. n elements for the mean and n
            # for the
            output_dim = distr_dim + distr_dim * (distr_dim + 1) // 2
            idxs = torch.tril_indices(distr_dim, distr_dim)

            self.register_buffer('row_idxs', idxs[0, :])
            self.register_buffer('col_idxs', idxs[1, :])
        else:
            raise ValueError(
                f'Unknown covariance type: {self.covariance_type}')

        self.net = build_network(network_cfg, input_dim, output_dim)
        logger.info(self.net)

    def _tensor_to_mean_cov(self, tensor: Tensor) -> Dict[str, Tensor]:
        mean = tensor[:, :self.distr_dim]
        cov_elements = tensor[:, self.distr_dim:]

        B, _ = tensor.shape
        L = torch.zeros([B, self.distr_dim, self.distr_dim],
                        dtype=mean.dtype, device=mean.device)

        if self.covariance_type == 'diagonal':
            L[:, self.row_idxs, self.col_idxs] = self.scale_func(cov_elements)
        elif self.covariance_type == 'tril':
            L[:, self.row_idxs, self.col_idxs] = cov_elements

        return {'mean': mean, 'L': L}

    def sample(
        self,
        N: int,
        cond: Tensor,
        mean: Optional[Tensor] = None,
        L: Optional[Tensor] = None,
    ) -> Tensor:
        if mean is None and L is None:
            param_dict = self._tensor_to_mean_cov(self.net(cond))
            mean = param_dict['mean']
            L = param_dict['L']

        B = len(cond)
        z = torch.randn([B, N, self.distr_dim],
                        dtype=cond.dtype, device=cond.device)

        # S = mu + L * z
        samples = mean.reshape(B, 1, -1) + torch.einsum('bmn,bsn->bsm', [L, z])

        return samples

    def samples_and_loglikelihood(
        self,
        N: int,
        cond: Tensor,
        values: Tensor,
        **kwargs,
    ) -> Dict[str, Tensor]:
        param_dict = self._tensor_to_mean_cov(self.net(cond))

        samples = None
        if N > 0:
            samples = self.sample(N, cond, mean=param_dict['mean'],
                                  L=param_dict['L'])

        ll = self.neg_log_likelihood(
            cond, values, mean=param_dict['mean'], L=param_dict['L'],
        )

        return {
            'samples': samples,
            'neg_log_likelihood': ll,
            **param_dict,
        }

    def neg_log_likelihood(
        self,
        cond: Tensor,
        values: Tensor,
        mean: Optional[Tensor] = None,
        L: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        ''' Computes the log-likelihood for the given samples
        '''
        if mean is None and L is None:
            param_dict = self._tensor_to_mean_cov(self.net(cond))
            mean = param_dict['mean']
            L = param_dict['L']

        inv_L = torch.inverse(L)

        L_diag = torch.diagonal(L, dim1=1, dim2=2)

        diff = values - mean

        prec = torch.matmul(inv_L.transpose(1, 2), inv_L)

        # Should be B-dimensional
        nll = 0.5 * (
            self.distr_dim * math.log(2 * math.pi) +
            2 * (L_diag.sum(dim=-1)).log() +
            (diff * torch.einsum('bmn,bn->bm', [prec, diff])).sum(dim=-1)
        )

        return nll

    def forward(
        self,
        cond: Tensor,
        values: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        param_dict = self._tensor_to_mean_cov(self.net(cond))
        output = dict(**param_dict)

        if values is not None:
            output['neg_log_likelihood'] = self.neg_log_likelihood(
                cond, values),

        return output


class ConditionalAffineCoupling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        network_cfg: DictConfig,
        scale=True,
        scale_func: str = 'softplus',
    ) -> None:
        super(ConditionalAffineCoupling, self).__init__()

        self.scale = scale
        self.scale_func, _ = build_scale_foo(scale_func)

        self.dim = input_dim // 2

        self.network = build_network(
            network_cfg, cond_dim, self.dim + scale * self.dim)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self._forward(x, context=context, rev=False)

    def inverse(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self._forward(x, context=context, rev=True)

    def _forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        rev: bool = False,
        jac: bool = True
    ) -> Tuple[Tensor, Tensor]:
        assert context is not None, (
            'Context cannot be none for ConditionalAffineCoupling')
        parameters = self.network(context)
        transl = parameters[:, :self.dim]

        if self.scale:
            scale = self.scale_func(parameters[:, self.dim:])
        else:
            scale = torch.ones_like(transl)

        top = x[:, :self.dim]
        bottom = x[:, :self.dim]

        logabsdet = scale.log().sum(dim=1)

        if rev:
            bottom_transf = (bottom - transl) / scale
            y = torch.cat([top, bottom_transf], dim=1)
            return y, -logabsdet
        else:
            bottom_transf = scale * bottom + transl
            y = torch.cat([top, bottom_transf], dim=1)

            return y, logabsdet


FLOW_NORMS = {
    'actnorm': transforms.ActNorm,
    'batch-norm': transforms.BatchNorm,
}

FLOW_PERMS = {
    'lu-linear': transforms.LULinear,
    'linear': transforms.NaiveLinear,
    'random': transforms.RandomPermutation,
}

FLOW_COUPLING = {
    'conditional-affine': ConditionalAffineCoupling,
    'conditional-additive':
    lambda *args, **kwargs: ConditionalAffineCoupling(
        *args, scale=False, **kwargs),
}


class FlowRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        distr_dim: int,
        scale_func: str = 'softplus',
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super(FlowRegressor, self).__init__()

        self.distr_dim = distr_dim

        prob_cfg = cfg.get('probabilistic', {})
        flow_cfg = prob_cfg.get('flow', {})
        network_cfg = cfg.get('network', {})

        self.scale_func = build_scale_foo(scale_func)

        num_blocks = flow_cfg.get('num_blocks', 4)

        norm_type = flow_cfg.get('norm_type', 'actnorm')
        perm_type = flow_cfg.get('perm_type', 'lu-linear')
        coupling_type = flow_cfg.get('coupling_type', 'lulinear')

        # self.blocks = nn.ModuleList()
        blocks = []
        for k in range(num_blocks):
            norm = FLOW_NORMS[norm_type](distr_dim)
            blocks.append(norm)

            perm_layer = FLOW_PERMS[perm_type](distr_dim)
            blocks.append(perm_layer)

            coupling_layer = FLOW_COUPLING[coupling_type](
                distr_dim, input_dim, network_cfg=network_cfg)
            blocks.append(coupling_layer)

        transform = transforms.CompositeTransform(blocks)

        # Define a base distribution.
        base_distribution = distributions.StandardNormal(shape=[distr_dim])

        # Combine into a flow.
        self.flow = flows.Flow(transform=transform,
                               distribution=base_distribution)

    def sample(
        self,
        N: int,
        cond: Tensor,
    ) -> Tensor:

        B = len(cond)
        # Sample from the distribution
        samples = self.flow.sample(N, context=None, batch_size=B)
        # logger.info(f'{B}, {N}')
        # logger.info(samples.shape)
        # sys.exit(0)

        return samples

    def samples_and_loglikelihood(
        self,
        N: int,
        cond: Tensor,
        values: Tensor,
        return_mean: bool = False,
    ) -> Dict[str, Tensor]:

        B = len(cond)
        # _, logabsdet = self.flow._transform(
        #     x.view(batch_size, -1), context=cond)
        # log_prob = self.flow._distribution.log_prob(u, context=cond)

        noise, _ = self.flow._distribution.sample_and_log_prob(
            N, context=cond
        )
        noise = noise.reshape(B * N, -1)
        if return_mean:
            mean_latent = torch.zeros(
                [B, self.distr_dim], dtype=cond.dtype, device=cond.device)
            noise = torch.cat([mean_latent, noise], dim=0)

        samples, _ = self.flow._transform.inverse(
            noise, context=cond.reshape(B, 1, -1).expand(-1, N + return_mean, -1).reshape(
                B * (N + return_mean), -1))

        nll = self.neg_log_likelihood(cond, values)
        output = {'neg_log_likelihood': nll}

        if return_mean:
            samples = samples.reshape(B, N + return_mean, -1)
            output['mean'] = samples[:, 0]
            output['samples'] = samples[:, 1:]
        else:
            output['samples'] = samples
        # for key, val in output.items():
        #     logger.info(f'{key}: {val.shape}')

        return output

    def neg_log_likelihood(
        self,
        cond: Tensor,
        values: Tensor,
    ) -> Dict[str, Tensor]:
        ''' Computes the log-likelihood for the given samples
        '''

        noise, logabsdet = self.flow._transform(values, context=cond)
        log_prob = self.flow._distribution.log_prob(noise, context=cond)
        return -(log_prob + logabsdet)

    def forward(
        self,
        cond: Tensor,
        values: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        B = len(cond)
        noise = torch.zeros([B, self.distr_dim], dtype=cond.dtype,
                            device=cond.device)

        mean, _ = self.flow._transform.inverse(noise, context=cond)
        # logger.info(f'Noise {noise.shape} -> mean {mean.shape}')

        output = {'mean': mean}
        if values is not None:
            output['neg_log_likelihood'] = self.neg_log_likelihood(
                cond, values),

        return output


def build_distr_regressor(
    cfg: DictConfig,
    input_dim: int,
    distr_dim: int,
) -> Union[MultiVariateNormalRegressor, FlowRegressor]:
    ''' Builds a probabilistic regressor
    '''
    prob_cfg = cfg.get('probabilistic', {})
    prob_type = prob_cfg.get('type', 'gaussian')

    if prob_type == 'gaussian' or prob_type == 'multivariate-normal':
        return MultiVariateNormalRegressor(
            input_dim, distr_dim, cfg=cfg)
    elif prob_type == 'flow':
        return FlowRegressor(input_dim, distr_dim, cfg=cfg)
    else:
        raise ValueError(f'Unknown distribution predictor type: {type}')
