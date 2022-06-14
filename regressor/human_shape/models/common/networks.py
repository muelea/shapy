from typing import Optional, Tuple, List, Union, Dict
import sys

import math

import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..nnutils import init_weights

from loguru import logger

from human_shape.utils import (
    CN, Tensor, Array, IntList, IntTuple, FloatList, FloatTuple, TensorList)

CONV_DIM_DICT = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d
}
TRANSPOSE_CONV_DIM_DICT = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d
}
BN_DIM_DICT = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


def build_activation(
    activ_cfg
) -> Union[nn.ReLU, nn.LeakyReLU, nn.PReLU]:
    ''' Builds activation functions
    '''
    if activ_cfg is None:
        return None
    activ_type = activ_cfg.get('type', 'relu')
    inplace = activ_cfg.get('inplace', False)
    if activ_type == 'relu':
        return nn.ReLU(inplace=inplace)
    elif activ_type == 'leaky-relu':
        leaky_relu_cfg = activ_cfg.get('leaky_relu', {})
        return nn.LeakyReLU(inplace=inplace, **leaky_relu_cfg)
    elif activ_type == 'prelu':
        prelu_cfg = activ_cfg.get('prelu', {})
        return nn.PReLU(inplace=inplace, **prelu_cfg)
    elif activ_type == 'none':
        return None
    else:
        raise ValueError(f'Unknown activation type: {activ_type}')


def build_norm_layer(
    input_dim: int,
    norm_cfg: Dict,
    dim: int = 1
) -> nn.Module:
    ''' Builds normalization modules
    '''
    if norm_cfg is None:
        return None
    norm_type = norm_cfg.get('type', 'bn')
    if norm_type == 'bn' or norm_type == 'batch-norm':
        bn_cfg = norm_cfg.get('batch_norm', {})
        if dim in BN_DIM_DICT:
            return BN_DIM_DICT[dim](input_dim, **bn_cfg)
        else:
            raise ValueError(f'Wrong dimension for BN: {dim}')
    elif norm_type == 'ln' or norm_type == 'layer-norm':
        layer_norm_cfg = norm_cfg.get('layer_norm', {})
        return nn.LayerNorm(input_dim, **layer_norm_cfg)
    elif norm_type == 'gn':
        group_norm_cfg = norm_cfg.get('group_norm', {})
        return nn.GroupNorm(num_channels=input_dim, **group_norm_cfg)
    elif norm_type.lower() == 'none':
        return None
    else:
        raise ValueError(f'Unknown normalization type: {norm_type}')


def build_rnn_cell(
    input_size: int,
    rnn_type='lstm',
    hidden_size=1024,
    bias=True
) -> Union[nn.LSTMCell, nn.GRUCell]:
    if rnn_type == 'lstm':
        return nn.LSTMCell(input_size, hidden_size=hidden_size, bias=bias)
    elif rnn_type == 'gru':
        return nn.GRUCell(input_size, hidden_size=hidden_size, bias=bias)
    else:
        raise ValueError(f'Unknown RNN type: {rnn_type}')


def build_rnn_stack(cfg, input_size: int, output_dim: int):
    num_layers = cfg.get('num_layers', 1)
    dropout = cfg.get('dropout', 0.0)
    rnn_type = cfg.get('type', 'lstm')
    rnn_cfg = cfg.get(rnn_type, {})
    if rnn_type == 'lstm':
        rnn_cell_type = nn.LSTMCell
    elif rnn_type == 'gru':
        rnn_cell_type = nn.GRUCell
    else:
        raise ValueError(f'Unknown RNN type: {rnn_type}')

    layers = []
    input_dim = input_size
    for ii in range(num_layers):
        rnn_cell = rnn_cell_type(cfg, input_dim)
        input_dim = cfg
        layers.append(rnn_cell)

    return MultiLayerRNN(layers, dropout=dropout)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    @staticmethod
    def from_bn(module: nn.BatchNorm2d):
        ''' Initializes a frozen batch norm module from a batch norm module
        '''
        dim = len(module.weight.data)

        frozen_module = FrozenBatchNorm2d(dim)
        frozen_module.weight.data = module.weight.data

        missing, not_found = frozen_module.load_state_dict(
            module.state_dict(), strict=False)
        return frozen_module

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    def forward(self, x: Tensor) -> Tensor:
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            False)


class MultiLayerRNNCell(nn.Module):
    ''' A multi-layer RNN cell

        Implements a multi-layer RNN cell with dropout. Each RNN layer receives
        the hidden state of its parent as input, after the application of
        dropout.
        Args:
            cfg: A dict-like object that contains the configuration arguments
            for each layer of the RNN
            input_size: The dimension of the input tensor
            output_dim: The dimension of the output tensor

        Inputs: input
        Outputs: output
    '''

    def __init__(
        self,
        cfg: DictConfig,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0, **kwargs
    ) -> None:
        super(MultiLayerRNNCell, self).__init__()
        self.dropout = dropout

        layer_dims = cfg.get('layer_dims', [1024])
        dropout = cfg.get('dropout', 0.0)
        rnn_type = cfg.get('type', 'lstm')
        bias = cfg.get('bias', True)
        learn_state = cfg.get('learn_state', False)

        if rnn_type == 'lstm':
            rnn_cell_type = nn.LSTMCell
        elif rnn_type == 'gru':
            rnn_cell_type = nn.GRUCell
        else:
            raise ValueError(f'Unknown RNN type: {rnn_type}')

        hidden_size = layer_dims[0]
        self.learn_state = learn_state
        init_type = cfg.get('init_type', 'zero')
        self.init_type = init_type
        if not learn_state:
            self.hidden_state = None
        else:
            if rnn_type == 'lstm':
                num_states = 2
            else:
                num_states = 1
            INIT_FUNCS = {
                'zeros': torch.zeros,
                'randn': torch.randn,
            }
            assert init_type in INIT_FUNCS
            shape = (1, hidden_size)
            self.hidden_state = nn.ParameterList()
            for n in range(num_states):
                self.hidden_state.append(nn.Parameter(
                    INIT_FUNCS[init_type](shape, dtype=torch.float32)))

        # Construct the RNN layer
        self.rnn_list = nn.ModuleList()
        curr_input_dim = input_dim
        for ii, hidden_size in enumerate(layer_dims):
            self.rnn_list.append(rnn_cell_type(
                curr_input_dim, hidden_size=hidden_size,
                bias=bias))
            curr_input_dim = hidden_size

        self.output = MLP(curr_input_dim, output_dim, **cfg.get('mlp'))
        self.input_dim = input_dim
        self.output_dim = output_dim
        logger.info(self)

    def extra_repr(self):
        msg = [
            f'Input -> Output: ({self.input_dim}) -> ({self.output_dim})',
            f'Learn initial hidden state: {self.learn_state}',
            (f'Learned state init func: {self.init_type}' if self.learn_state
             else ''),
            f'Dropout: {self.dropout}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Union[Tensor, TensorList]]:
        ''' Forward pass
        '''
        # Extract the batch_size
        batch_size = x.shape[0]
        # Expand the shape of the state to match the current batch
        state = None
        if self.hidden_state is not None:
            state = [s.expand(batch_size, -1) for s in self.hidden_state]
        for ii in range(len(self.rnn_list)):
            module = self.rnn_list[ii]
            x, state = module(x, state)
            # Extract the correct state vector. LSTMs return a tuple, while
            # GRUs return a single tensor.
            if isinstance(state, (tuple, list)):
                x = state[0]
            elif torch.is_tensor(state):
                x = state
            # Apply dropout to the hidden state
            if ii < len(self.rnn_list) - 1 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        # Apply the output layer
        return self.output(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: IntList = None,
        activation: Optional[DictConfig] = None,
        normalization: Optional[DictConfig] = None,
        dropout: float = 0.0,
        gain: float = 0.01,
        preactivated: bool = False,
        flatten: bool = True,
        **kwargs
    ) -> None:
        ''' Simple MLP layer
        '''
        super(MLP, self).__init__()
        if layers is None:
            layers = []
        self.flatten = flatten
        self.input_dim = input_dim
        self.output_dim = output_dim

        if activation is None:
            activation = {}
        if normalization is None:
            normalization = {}

        logger.info(f'Output layer gain: {gain}')

        curr_input_dim = input_dim
        self.num_layers = len(layers)
        self.blocks = []
        for layer_idx, layer_dim in enumerate(layers):
            activ = build_activation(activation)
            norm_layer = build_norm_layer(
                layer_dim, norm_cfg=normalization, dim=1)
            bias = norm_layer is None

            linear = nn.Linear(curr_input_dim, layer_dim, bias=bias)
            curr_input_dim = layer_dim

            layer = []
            if preactivated:
                if norm_layer is not None:
                    layer.append(norm_layer)

                if activ is not None:
                    layer.append(activ)

                layer.append(linear)

                if dropout > 0.0:
                    layer.append(nn.Dropout(dropout))
            else:
                layer.append(linear)

                if activ is not None:
                    layer.append(activ)

                if norm_layer is not None:
                    layer.append(norm_layer)

                if dropout > 0.0:
                    layer.append(nn.Dropout(dropout))

            block = nn.Sequential(*layer)
            self.add_module('layer_{:03d}'.format(layer_idx), block)
            self.blocks.append(block)

        self.output_layer = nn.Linear(curr_input_dim, output_dim)
        init_weights(
            self.output_layer, gain=gain,
            activ_type=activation.get('type', 'relu'),
            init_type='xavier', distr='uniform')

    def extra_repr(self):
        msg = [
            f'Input ({self.input_dim}) -> Output ({self.output_dim})',
            f'Flatten: {self.flatten}',
        ]

        return '\n'.join(msg)

    def forward(self, module_input: Tensor, **kwargs) -> Tensor:
        batch_size = module_input.shape[0]
        # Flatten all dimensions
        curr_input = module_input
        if self.flatten:
            curr_input = curr_input.view(batch_size, -1)
        for block in self.blocks:
            curr_input = block(curr_input)
        return self.output_layer(curr_input)


class FCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        #  output_dim: int,
        layers: IntList = None,
        activation: Optional[DictConfig] = None,
        normalization: Optional[DictConfig] = None,
        dropout: float = 0.0,
        gain: float = 0.01,
        preactivated: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        **kwargs
    ) -> None:
        ''' Simple FCN layer
        '''
        super(FCN, self).__init__()
        if layers is None:
            layers = []
        self.input_dim = input_dim

        curr_input_dim = input_dim
        self.num_layers = len(layers)
        self.blocks = []
        for layer_idx, layer_dim in enumerate(layers):
            activ = build_activation(activation)
            norm_layer = build_norm_layer(
                layer_dim, norm_cfg=normalization, dim=2)
            bias = norm_layer is None

            linear = nn.Conv2d(curr_input_dim, layer_dim,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias)
            curr_input_dim = layer_dim

            layer = []
            if preactivated:
                if norm_layer is not None:
                    layer.append(norm_layer)

                if activ is not None:
                    layer.append(activ)

                layer.append(linear)

                if dropout > 0.0:
                    layer.append(nn.Dropout(dropout))
            else:
                layer.append(linear)

                if activ is not None:
                    layer.append(activ)

                if norm_layer is not None:
                    layer.append(norm_layer)

                if dropout > 0.0:
                    layer.append(nn.Dropout(dropout))

            block = nn.Sequential(*layer)
            self.add_module('layer_{:03d}'.format(layer_idx), block)
            self.blocks.append(block)

        #  self.output_layer = nn.Linear(curr_input_dim, output_dim)
        #  init_weights(
            #  self.output_layer, gain=gain,
            #  activ_type=activation.get('type', 'relu'),
            #  init_type='xavier', distr='uniform')

    def extra_repr(self):
        msg = [
            f'Input ({self.input_dim})',
        ]

        return '\n'.join(msg)

    def forward(self, module_input: Tensor, **kwargs) -> Tensor:
        # Flatten all dimensions
        curr_input = module_input
        for block in self.blocks:
            curr_input = block(curr_input)
        return curr_input
        #  return self.output_layer(curr_input)


class IterativeRegression(nn.Module):
    def __init__(
        self,
        module: Union[MLP, MultiLayerRNNCell],
        mean_param: Tensor,
        num_stages: int = 1,
        append_params: bool = True,
        learn_mean: bool = False,
        detach_mean: bool = False,
        dim: int = 1,
        **kwargs
    ) -> None:
        super(IterativeRegression, self).__init__()
        logger.info(f'Building iterative regressor with {num_stages} stages')

        self.module = module
        self._num_stages = num_stages
        self.dim = dim
        self.append_params = append_params
        self.detach_mean = detach_mean
        self.learn_mean = learn_mean

        if learn_mean:
            self.register_parameter(
                'mean_param', nn.Parameter(mean_param, requires_grad=True))
        else:
            self.register_buffer('mean_param', mean_param)

    def get_mean(self):
        return self.mean_param.clone()

    @property
    def num_stages(self):
        return self._num_stages

    def extra_repr(self):
        msg = [
            f'Num stages = {self.num_stages}',
            f'Concatenation dimension: {self.dim}',
            f'Detach mean: {self.detach_mean}',
            f'Learn mean: {self.learn_mean}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        features: Tensor,
        cond: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[TensorList, TensorList]:
        ''' Computes deltas on top of condition iteratively

            Parameters
            ----------
                features: torch.Tensor
                    Input features computed by a NN backbone
                cond: Condition vector used as the initial point of the
                    iterative regression method.

            Returns
            -------
                parameters: List[torch.Tensor]
                    A list of tensors, where each element corresponds to a
                    different stage
                deltas: List[torch.Tensor]
                    A list of tensors, where each element corresponds to a
                    the estimated offset at each stage
        '''
        batch_size = features.shape[0]
        expand_shape = [batch_size] + [-1] * len(features.shape[1:])

        parameters = []
        deltas = []
        module_input = features

        if cond is None:
            cond = self.mean_param.expand(*expand_shape).clone()

        # Detach mean
        if self.detach_mean:
            cond = cond.detach()

        if self.append_params:
            assert features is not None, (
                'Features are none even though append_params is True')

            module_input = torch.cat([
                module_input,
                cond],
                dim=self.dim)
        deltas.append(self.module(module_input))
        num_params = deltas[-1].shape[1]
        parameters.append(cond[:, :num_params].clone() + deltas[-1])

        for stage_idx in range(1, self.num_stages):
            module_input = torch.cat(
                [features, parameters[stage_idx - 1]], dim=-1)
            params_upd = self.module(module_input)
            parameters.append(parameters[stage_idx - 1] + params_upd)

        return parameters, deltas


class RNNIterativeRegressor(IterativeRegression):
    def __init__(self, *args, **kwargs):
        super(RNNIterativeRegressor, self).__init__(*args, **kwargs)

    def forward(
        self,
        features: Tensor,
        cond: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[TensorList, TensorList]:
        ''' Computes deltas on top of condition iteratively

            Parameters
            ----------
                features: torch.Tensor
                    Input features computed by a NN backbone
                cond: Condition vector used as the initial point of the
                    iterative regression method.

            Returns
            -------
                parameters: List[torch.Tensor]
                    A list of tensors, where each element corresponds to a
                    different stage
                deltas: List[torch.Tensor]
                    A list of tensors, where each element corresponds to a
                    the estimated offset at each stage
        '''
        batch_size = features.shape[0]
        expand_shape = [batch_size] + [-1] * len(features.shape[1:])

        parameters, deltas = [], []
        if cond is None:
            cond = self.mean_param.expand(*expand_shape).clone()
        # Detach mean
        if self.detach_mean:
            cond = cond.detach()

        # Build the input vector
        if self.append_params:
            assert cond is not None, (
                'Condition is none even though append_params is True')
            module_input = torch.cat([features, cond], dim=self.dim)
        else:
            module_input = features

        for stage_idx in range(self.num_stages):
            params_upd = self.module(module_input)
            deltas.append(params_upd)

            if stage_idx == 0:
                next_params = cond + params_upd
            else:
                next_params = cond + parameters[stage_idx - 1]
            parameters.append(next_params)
            # Create the input for the next iteration
            if self.append_params:
                assert cond is not None, (
                    'Condition is none even though append_params is True')
                module_input = torch.cat([features, cond], dim=self.dim)
            else:
                module_input = features

        return parameters, deltas


class NonLocalBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 dim=2,
                 reduction: int = 2) -> None:
        ''' Implements the self-attention operation for non local feature
            pooling

            For more information see:
            @inproceedings{wang2018non,
              title={Non-local neural networks},
              author={Wang, Xiaolong and Girshick, Ross and Gupta, Abhinav and
                  He, Kaiming},
              booktitle={Proceedings of the IEEE Conference on Computer Vision
                  and Pattern Recognition},
              pages={7794--7803},
              year={2018}
            }
        '''

        super(NonLocalBlock, self).__init__()

        if dim == 2:
            conv = nn.Conv2d
        elif dim == 3:
            conv = nn.Conv3d

        self.dim = dim
        self.reduction = reduction

        # Bottlenecks used to reduce the channel dimensions
        self.conv1 = conv(
            channels, channels // reduction,
            kernel_size=1, stride=1,
            padding=0)
        self.conv2 = conv(
            channels // reduction, channels,
            kernel_size=1, stride=1,
            padding=0)

        scale = nn.Parameter(torch.zeros(
            [1, channels] + [1] * dim, dtype=torch.float32))
        self.register_parameter('scale', scale)

    def extra_repr(self) -> str:
        msg = [
            f'Dim: {self.dim}',
            f'Scale: {self.scale.shape}'
        ]
        return '\n'.join(msg)

    def forward(self, x: Tensor) -> Tensor:
        B, C = x.shape[:2]

        # y is Bx(THW)x(C // 2) if dim == 3 else y is Bx(HW)x(C // 2) if dim ==
        # 2
        y = self.conv1(x).view(B, C // self.reduction, -1).permute(0, 2, 1)

        # Should be Bx(THW)x(THW)
        attention = torch.einsum('bmc,bcn->bmn', [y, y.permute(0, 2, 1)])
        attention = F.softmax(attention, dim=-1)
        y = torch.matmul(attention, y).view(B, -1, *x.shape[2:])

        return self.conv2(y) * self.scale + x


def build_regressor(
    network_cfg,
    input_dim, output_dim, param_mean
) -> Union[MLP, IterativeRegression]:
    regressor_type = network_cfg.get('type', 'mlp')
    logger.info(f'Building: {regressor_type}')
    if regressor_type == 'mlp':
        regressor_cfg = network_cfg.get(regressor_type, {})
        regressor = MLP(input_dim, output_dim, **regressor_cfg)
        return regressor, 1
    elif regressor_type == 'iterative-mlp':
        mlp_cfg = network_cfg.get('mlp', {})

        append_params = network_cfg.get('append_params', True)
        # Build the MLP used to regress the parameters
        regressor = MLP(input_dim + append_params * param_mean.numel(),
                        output_dim, **mlp_cfg)
        # Build the iterative regression object
        iterative_regressor = IterativeRegression(
            regressor, param_mean, **network_cfg)
        return iterative_regressor, iterative_regressor.num_stages
    elif regressor_type == 'iterative-rnn':
        rnn_cfg = network_cfg.get('rnn', {})

        append_params = network_cfg.get('append_params', True)
        # Build the MLP used to regress the parameters
        regressor = MultiLayerRNNCell(
            rnn_cfg,
            input_dim + append_params * param_mean.numel(), output_dim,
            **rnn_cfg)
        # Build the iterative regression object
        iterative_regressor = RNNIterativeRegressor(
            regressor, param_mean, **network_cfg)
        return iterative_regressor, iterative_regressor.num_stages
    else:
        raise ValueError(f'Unknown regressor type: {regressor_type}')


def build_network(
    input_dim: int,
    output_dim: int,
    cfg: DictConfig,
) -> Union[MLP]:
    type = cfg.get('type', 'mlp')
    if type == 'mlp':
        mlp_cfg = cfg.get('mlp', {})
        return MLP(input_dim, output_dim, **mlp_cfg)
    else:
        raise ValueError(f'Unknown network type: {type}')
