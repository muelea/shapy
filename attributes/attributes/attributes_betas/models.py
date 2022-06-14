from typing import Optional, Tuple, List, Union, Dict

import sys
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
import torch.nn.init as nninit
from omegaconf import DictConfig

from loguru import logger
from .polynomial import Polynomial
from attributes.utils.typing import Tensor, IntList, TensorList


BN_DIM_DICT = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


def build_activation(
    activ_cfg
) -> Union[nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU]:
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
        return nn.PReLU(**prelu_cfg)
    elif activ_type == 'elu':
        elu_cfg = activ_cfg.get('elu', {})
        return nn.ELU(inplace=inplace, **elu_cfg)
    elif activ_type == 'none':
        return None
    else:
        raise ValueError(f'Unknown activation type: {activ_type}')


def build_normalization(
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
    elif norm_type == 'gn' or norm_type == 'group-norm':
        group_norm_cfg = norm_cfg.get('group_norm', {})
        return nn.GroupNorm(num_channels=input_dim, **group_norm_cfg)
    elif norm_type.lower() == 'none':
        return None
    else:
        raise ValueError(f'Unknown normalization type: {norm_type}')


class FCNormActiv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        activation: Optional[DictConfig] = None,
        normalization: Optional[DictConfig] = None,
        dropout: float = 0.0,
    ) -> None:
        super(FCNormActiv, self).__init__()

        activ = build_activation(activation)

        norm_layer = build_normalization(
            output_dim, norm_cfg=normalization, dim=1)
        bias = norm_layer is None and bias
        self.fc = nn.Linear(input_dim, output_dim, bias=bias)
        self.norm_layer = norm_layer
        self.activ = activ

        self.dropout = dropout
        if dropout > 0.0:
            self.drop_layer = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.activ is not None:
            x = self.activ(x)
        if self.dropout > 0:
            x = self.drop_layer(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: Optional[IntList] = None,
        activation: Optional[DictConfig] = None,
        normalization: Optional[DictConfig] = None,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ) -> None:
        ''' Simple MLP layer
        '''
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if activation is None:
            activation = {}
        if normalization is None:
            normalization = {}

        curr_input_dim = input_dim
        self.num_layers = len(layers)
        self.layers = nn.ModuleList()
        for layer_idx, layer_dim in enumerate(layers):
            self.layers.append(
                FCNormActiv(curr_input_dim, layer_dim, bias=bias,
                            normalization=normalization, activation=activation,
                            dropout=dropout)
            )
            curr_input_dim = layer_dim

        self.output_layer = nn.Linear(curr_input_dim, output_dim)

    def extra_repr(self):
        msg = [
            f'Input ({self.input_dim}) -> Output ({self.output_dim})',
        ]
        return '\n'.join(msg)

    def forward(self, module_input: Tensor, **kwargs) -> Tensor:
        # Flatten all dimensions
        curr_input = module_input
        for ii, layer in enumerate(self.layers):
            curr_input = layer(curr_input)
        return self.output_layer(curr_input)


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_features: int = 256,
        activation: Optional[DictConfig] = None,
        normalization: Optional[DictConfig] = None,
        dropout: float = 0.0,
    ):
        super(BasicBlock, self).__init__()
        if normalization is None:
            normalization = {}
        if activation is None:
            activation = {}

        self.norm1 = build_normalization(hidden_features, normalization, dim=1)
        self.linear1 = nn.Linear(
            input_dim, hidden_features, bias=self.norm1 is not None)
        self.act = build_activation(activation)
        self.norm2 = build_normalization(output_dim, normalization, dim=1)
        self.linear2 = nn.Linear(
            hidden_features, output_dim, bias=self.norm2 is not None)
        self.downsample = None
        if input_dim != output_dim:
            norm = build_normalization(output_dim, normalization, dim=1)
            downsample = [nn.Linear(
                input_dim, output_dim, bias=norm is not None)]
            if norm is not None:
                downsample.append(norm)
            self.downsample = nn.Sequential(*downsample)

        self.dropout = dropout
        if dropout > 0.0:
            self.drop_layer = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.act(out)
        if self.dropout > 0:
            out = self.drop_layer(out)

        out = self.linear2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        out = self.act(out)
        if self.dropout > 0:
            out = self.drop_layer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim: int, output_dim: int,
        layers: IntList = (256, 256),
        activation: Optional[DictConfig] = None,
        normalization: Optional[DictConfig] = None,
        dropout: float = 0.0,
        proj_layer=True,
        **kwargs,
    ) -> None:
        super(ResNet, self).__init__()

        blocks = []

        self.proj_layer = proj_layer
        curr_in_dim = input_dim
        if self.proj_layer:
            self.projection = FCNormActiv(curr_in_dim, layers[0],
                                          normalization=normalization, activation=activation,
                                          dropout=dropout)
            curr_in_dim = layers[0]

        for ii, feats in enumerate(layers):
            blocks.append(
                BasicBlock(curr_in_dim, feats, hidden_features=feats,
                           activation=activation, normalization=normalization,
                           dropout=dropout)
            )
            curr_in_dim = feats

        blocks.append(nn.Linear(curr_in_dim, output_dim))
        self.network = nn.Sequential(*blocks)

    def forward(self, x, context=None):
        if self.proj_layer:
            x = self.projection(x)
        return self.network(x)


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        moe_cfg: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        ''' Mixture of Experts architecture
        '''
        super(MixtureOfExperts, self).__init__()

        num_experts = moe_cfg.get('num_experts', 8)
        self.num_experts = num_experts

        network_cfg = moe_cfg.get('network', {})

        self.gating = build_network(network_cfg, input_dim, num_experts)

        self.ffns = nn.ModuleList()
        for ii in range(num_experts):
            network = build_network(network_cfg, input_dim, output_dim)
            self.ffns.append(network)

    def extra_repr(self) -> str:
        msg = [
            f'Number of experts: {self.num_experts}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        module_input: Tensor,
    ):
        gating_weights = F.softmax(self.gating(module_input), dim=-1)

        ffn_outputs = []
        for ii, ffn in enumerate(self.ffns):
            ffn_outputs.append(ffn(module_input))

        ffn_outputs = torch.stack(ffn_outputs, dim=1)
        output = (gating_weights.unsqueeze(dim=-1) * ffn_outputs).sum(dim=1)

        return output


class MixtureOfInputExperts(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        input_moe_cfg: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        ''' Mixture of Experts architecture
        '''
        super(MixtureOfInputExperts, self).__init__()

        self.num_experts = input_dim

        network_cfg = input_moe_cfg.get('network', {})

        self.gating = build_network(network_cfg, input_dim, input_dim)

        self.ffns = nn.ModuleList()
        for ii in range(self.num_experts):
            network = build_network(network_cfg, 1, output_dim)
            self.ffns.append(network)

    def extra_repr(self) -> str:
        msg = [
            f'Number of experts: {self.num_experts}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        module_input: Tensor,
    ):
        gating_weights = F.softmax(self.gating(module_input), dim=-1)

        ffn_outputs = []
        for ii, ffn in enumerate(self.ffns):
            ffn_outputs.append(ffn(module_input[:, ii].unsqueeze(dim=-1)))

        ffn_outputs = torch.stack(ffn_outputs, dim=1)
        output = (gating_weights.unsqueeze(dim=-1) * ffn_outputs).sum(dim=1)

        return output


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
    ) -> None:
        super(MultiLayerRNNCell, self).__init__()

        self.dropout = cfg.get('dropout', 0.0)
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
        num_states = 2 if rnn_type == 'lstm' else 1
        self.num_states = num_states
        INIT_FUNCS = {
            'zeros': torch.zeros,
            'randn': torch.randn,
        }
        assert init_type in INIT_FUNCS
        shape = (1, hidden_size)
        if learn_state:
            self.hidden_state = nn.ParameterList()
            for n in range(num_states):
                self.hidden_state.append(nn.Parameter(
                    INIT_FUNCS[init_type](shape, dtype=torch.float32)))
        else:
            self.hidden_state = []
            for n in range(num_states):
                self.register_buffer(
                    f'state{n:02d}',
                    INIT_FUNCS[init_type](shape, dtype=torch.float32))

        # Construct the RNN layer
        self.rnn_list = nn.ModuleList()
        curr_input_dim = input_dim
        for ii, hidden_size in enumerate(layer_dims):
            self.rnn_list.append(rnn_cell_type(
                curr_input_dim, hidden_size=hidden_size,
                bias=bias))
            curr_input_dim = hidden_size

        self.output = nn.Linear(curr_input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

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
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Union[Tensor, TensorList]]:
        ''' Forward pass
        '''
        # Extract the batch_size
        batch_size = x.shape[0]
        # Expand the shape of the state to match the current batch
        if state is None:
            state = []
            for n in range(self.num_states):
                s = getattr(self, f'state{n:02d}').expand(batch_size, -1)
                state.append(s)

        output_states = []
        for ii, module in enumerate(self.rnn_list):
            state = module(x, *state)
            output_states.append(state)
            # Extract the correct state vector. LSTMs return a tuple, while
            # GRUs return a single tensor.
            if self.dropout > 0:
                if isinstance(state, (tuple, list)):
                    state[0] = F.dropout(
                        state[0], p=self.dropout, training=self.training)
                elif torch.is_tensor(state):
                    state = F.dropout(
                        state, p=self.dropout, training=self.training)

        if isinstance(state, (tuple, list)):
            output_hidden = output_states[-1][0]
        elif torch.is_tensor(state):
            output_hidden = output_states[-1]
        # Apply the output layer
        return self.output(output_hidden), output_states


class IterativeRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        iter_cfg: Optional[DictConfig] = None,
        param_mean: Optional[Tensor] = None,
        **kwargs
    ) -> None:
        super(IterativeRegressor, self).__init__()

        network_cfg = iter_cfg.get('network', {})

        append_params = iter_cfg.get('append_params', True)
        num_stages = iter_cfg.get('num_stages', 3)

        self.append_params = append_params
        self.num_stages = num_stages

        # Build the MLP used to regress the parameters
        self.regressor = MultiLayerRNNCell(
            network_cfg.get('rnn', {}),
            input_dim + append_params * output_dim, output_dim)

        if param_mean is None:
            param_mean = torch.zeros([output_dim], dtype=torch.float32)

        self.register_buffer('param_mean', param_mean)

    def forward(
        self,
        module_input: Tensor
    ) -> Tensor:
        B = len(module_input)

        cond = self.param_mean.unsqueeze(dim=0).expand(B, -1)

        if self.append_params:
            reg_input = torch.cat([module_input, cond], dim=-1)
        else:
            reg_input = module_input

        deltas, state = self.regressor(reg_input)
        # logger.info(f'Deltas 00: {deltas.shape}')
        # logger.info(f'State 00: {state.shape}')

        parameters = []

        parameters.append(cond.clone() + deltas)
        for stage_idx in range(1, self.num_stages):
            if self.append_params:
                reg_input = torch.cat([module_input, cond], dim=-1)
            else:
                reg_input = module_input
            deltas, state = self.regressor(reg_input, state=state)
            # logger.info(f'Deltas {stage_idx:02d}: {deltas.shape}')
            # logger.info(f'State {stage_idx:02d}: {state.shape}')
            parameters.append(parameters[-1] + deltas)

        return parameters[-1]


@torch.no_grad()
def weight_init(
    weights: Tensor,
    name: str = '',
    init_cfg: Optional[DictConfig] = None,
    activ_type: str = 'leaky-relu',
    lrelu_slope: float = 0.01,
    **kwargs
) -> None:
    if init_cfg is None:
        init_cfg = {}

    init_type = init_cfg.get('type', 'xavier')
    distr = init_cfg.get('distr', 'uniform')
    gain = init_cfg.get('gain', 1.0)

    logger.debug(
        'Initializing {} with {}_{}: gain={}', name, init_type, distr, gain)
    if init_type == 'xavier':
        if distr == 'uniform':
            nninit.xavier_uniform_(weights, gain=gain)
        elif distr == 'normal':
            nninit.xavier_normal_(weights, gain=gain)
        else:
            raise ValueError(f'Unknown distribution "{distr}" for Xavier init')
    elif init_type == 'kaiming':
        activ_type = activ_type.replace('-', '_')
        if distr == 'uniform':
            nninit.kaiming_uniform_(
                weights, a=lrelu_slope, mode='fan_in', nonlinearity=activ_type)
        elif distr == 'normal':
            nninit.kaiming_normal_(
                weights, a=lrelu_slope, mode='fan_in', nonlinearity=activ_type)
        else:
            raise ValueError(
                'Unknown distribution "{distr}" for Kaiming init')
    else:
        logger.warning(f'Unknown init type: {init_type}, leaving weights'
                       ' unchanged.')
    return


def build_network(
    network_cfg: DictConfig, input_dim: int, output_dim: int
) -> Union[MLP, ResNet, nn.Linear]:
    network_type = network_cfg.get('type', 'mlp')

    activ_cfg = {}

    if network_type == 'mlp':
        mlp_cfg = network_cfg.get('mlp', {})
        activ_cfg = mlp_cfg.get('activation', {})
        network = MLP(input_dim, output_dim, **mlp_cfg)
    elif network_type == 'resnet':
        resnet_cfg = network_cfg.get('resnet', {})
        activ_cfg = resnet_cfg.get('activation', {})
        network = ResNet(input_dim, output_dim, **resnet_cfg)
    elif network_type == 'moe' or network_type == 'mixture-of-experts':
        moe_cfg = network_cfg.get('moe', {})
        network = MixtureOfExperts(input_dim, output_dim, moe_cfg)
    elif network_type == 'imoe' or network_type == 'mixture-of-input-experts':
        input_moe_cfg = network_cfg.get('imoe', {})
        network = MixtureOfInputExperts(input_dim, output_dim, input_moe_cfg)
    elif network_type == 'iterative':
        iterative_cfg = network_cfg.get('iterative', {})
        network = IterativeRegressor(input_dim, output_dim, iterative_cfg)
    elif network_type == 'linear':
        network = nn.Linear(input_dim, output_dim)
    elif network_type == 'polynomial':
        polynomial_cfg = network_cfg.get('polynomial', {})
        network = Polynomial(input_dim, output_dim, **polynomial_cfg)
    elif network_type == 'simple':
        l1_size = int(input_dim - ((input_dim - output_dim) / 3))
        l2_size = int(input_dim - 2 * ((input_dim - output_dim) / 3))
        activ_cfg = {'type': 'relu'}
        simple_net = nn.Sequential(
            nn.Linear(input_dim, l1_size),
            nn.ReLU(),
            nn.Linear(l1_size, l2_size),
            nn.ReLU(),
            nn.Linear(l2_size, output_dim)
        )
        network = simple_net

    else:
        raise ValueError(f'Unknown network type: {network_type}')

    init_cfg = network_cfg.get('init', {})

    for name, module in network.named_modules():
        if 'linear' in name or 'fc' in name:
            weight_init(module.weight, name=name,
                        init_cfg=init_cfg,
                        activ_type=activ_cfg.get('type', 'relu'),
                        lrelu_slop=activ_cfg.get('leaky_relu', {}).get(
                            'negative_slope', 0.01)
                        )
    return network
