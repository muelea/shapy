from typing import Tuple, Optional
import sys
import os
from dataclasses import dataclass, field
from attributes.utils.constants import ATTRIBUTE_NAMES, ATTRIBUTE_NAMES_SYNTHETIC_DATA

import numpy as np
import argparse
from loguru import logger

from omegaconf import OmegaConf


@dataclass
class LeakyReLU:
    negative_slope: float = 0.01


@dataclass
class ELU:
    alpha: float = 1.0


@dataclass
class PReLU:
    num_parameters: int = 1
    init: float = 0.25


@dataclass
class Activation:
    type: str = 'relu'
    inplace: bool = False

    leaky_relu: LeakyReLU = LeakyReLU()
    prelu: PReLU = PReLU()
    elu: ELU = ELU()


@dataclass
class BatchNorm:
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True


@dataclass
class GroupNorm:
    num_groups: int = 32
    eps: float = 1e-05
    affine: bool = True


@dataclass
class Normalization:
    type: str = 'batch-norm'
    batch_norm: BatchNorm = BatchNorm()
    group_norm: GroupNorm = GroupNorm()


@dataclass
class MLP:
    layers: Tuple[int] = (256, 256)
    activation: Activation = Activation()
    normalization: Normalization = Normalization()
    dropout: float = 0.0
    bias: bool = True


@dataclass
class LSTM:
    bias: bool = True
    hidden_size: int = 1024


@dataclass
class GRU:
    bias: bool = True
    hidden_size: int = 1024


@dataclass
class RNN:
    type: str = 'gru'
    layer_dims: Tuple[int] = (1024,)
    init_type: str = 'randn'
    learn_mean: bool = False
    dropout: float = 0.0
    lstm: LSTM = LSTM()
    gru: GRU = GRU()


@dataclass
class ResNet:
    layers: Tuple[int] = (256, 256)
    activation: Activation = Activation()
    normalization: Normalization = Normalization()
    dropout: float = 0.0
    bias: bool = True


@dataclass
class MixtureOfExperts:
    num_experts: int = 8

    @dataclass
    class Network:
        type: str = 'mlp'
        mlp: MLP = MLP()
        resnet: ResNet = ResNet()
    network: Network = Network()


@dataclass
class IterativeRegressor:
    type: str = 'rnn'
    append_params: bool = True
    num_stages: int = 3
    detach_mean: bool = False
    learn_mean: bool = False

    @dataclass
    class Network:
        type: str = 'mlp'
        mlp: MLP = MLP()
        rnn: RNN = RNN()
    network: Network = Network()


@dataclass
class NetworkInit:
    type: str = 'xavier'
    distr: str = 'uniform'
    gain: float = 1.0


@dataclass
class Polynomial:
    degree: int = 2
    alpha: float = 0.0


@dataclass
class Network:
    type: str = 'mlp'
    polynomial: Polynomial = Polynomial()
    mlp: MLP = MLP()
    resnet: ResNet = ResNet()
    moe: MixtureOfExperts = MixtureOfExperts()
    imoe: MixtureOfExperts = MixtureOfExperts()
    iterative: IterativeRegressor = IterativeRegressor()

    init: NetworkInit = NetworkInit()


@dataclass
class FemaleAttributes:
    big: bool = True
    broad_shoulders: bool = True
    large_breasts: bool = True
    long_legs: bool = True
    long_neck: bool = True
    long_torso: bool = True
    muscular: bool = True
    pear_shaped: bool = True
    petite: bool = True
    short: bool = True
    short_arms: bool = True
    skinny_legs: bool = True
    slim_waist: bool = True
    tall: bool = True
    feminine: bool = True


@dataclass
class MaleAttributes:
    average: bool = True
    big: bool = True
    broad_shoulders: bool = True
    delicate_build: bool = True
    long_legs: bool = True
    long_neck: bool = True
    long_torso: bool = True
    masculine: bool = True
    muscular: bool = True
    rectangular: bool = True
    short: bool = True
    short_arms: bool = True
    skinny_arms: bool = True
    soft_body: bool = True
    tall: bool = True


@dataclass
class Measurements:
    # best guess
    height_bg: bool = False
    weight_bg: bool = False
    # ground truth
    height_gt: bool = False
    weight_gt: bool = False
    chest: bool = False
    waist: bool = False
    hips: bool = False


@dataclass
class MultivariateNormal:
    covariance: str = 'diagonal'


@dataclass
class Coupling:
    type: str = 'conditional-additive'
    scale_func: str = 'softplus'


@dataclass
class Flow:
    norm_type: str = 'actnorm'
    perm_type: str = 'lu-linear'
    coupling_type: str = 'conditional-additive'


@dataclass
class Probabilistic:
    type: str = 'gaussian'
    num_samples: int = 32
    noise_std: float = 0.0
    gaussian = MultivariateNormal = MultivariateNormal()
    flow: Flow = Flow()


@dataclass
class Regression:
    type: str = 'linear'
    alpha: float = 0.0
    kernel: str = 'linear'
    degree: int = 3
    use_loo_cross_val: bool = False
    # use to have the same setting as BodyTalk whw2s.
    # This means convert featues (weight -> sqrt(weight), height_m -> height_cm)
    # and adds gaussian noise to the test data.
    use_whw2s_setting: bool = False
    add_noise: bool = False


@dataclass
class Scheduler:
    type: str = 'exponential'
    interval: str = 'epoch'
    frequency: int = 1

    @dataclass
    class Exponential:
        gamma: float = 0.99
        last_epoch: int = -1
    exponential: Exponential = Exponential()


@dataclass
class Config:
    max_duration: float = 3600
    train: bool = True
    eval_val: bool = True
    eval_test: bool = False
    render_result: bool = True
    num_shape_comps: int = 10

    type: str = ''

    # Use the bodytalk pre-processing for measurements
    bodytalk_meas_preprocess: bool = False

    use_srb_noise: bool = False

    use_attr_noise: bool = False
    attr_noise_range: float = 0.2
    # Upper bound for the RMSE used to generated synthetic betas
    use_betas_noise: bool = False
    noise_rmse: float = 3
    lr: float = 1.0e-03
    weight_decay: float = 0.0

    use_scheduler: bool = False
    scheduler: Scheduler = Scheduler()

    edge_weight: float = 0.0
    height_weight: float = 0.0
    chest_weight: float = 0.0
    waist_weight: float = 0.0
    hips_weight: float = 0.0

    val_metric: str = 'v2v'
    v2v_weight: float = 0.0
    vertex_loss_metric: str = 'l2'
    v2v_hd_weight: float = 0.0
    betas_weight: float = 0.0
    betas_loss_metric: str = 'l2'
    ll_weight: float = 0.0

    output_dir: str = 'output'

    batch_size: int = 32
    max_epochs: int = 10000

    render_summary_steps: int = 50000

    ds_gender: str = ''
    model_gender: str = ''
    model_type: str = 'smplx'
    dataset: list = field(default_factory=lambda: ['caesar'])

    # Instead of the stored mean rating, compute it on the fly from
    # the raw ratings using samples
    sample_raw: bool = False
    num_samples: int = 10
    # Function used to aggregate the rating samples
    sample_agg_foo: str = 'mean'
    # Whether to normalize the input data
    normalize: bool = False

    checkpoint: str = ''

    # Maximum V2V error value in mm used for defining the bounds of the colormap
    max_val: float = 20.0
    cmap: str = 'viridis'
    align: bool = True

    # Select input features
    use_attributes: bool = True
    female_attributes: FemaleAttributes = FemaleAttributes()
    male_attributes: MaleAttributes = MaleAttributes()
    use_measurements: bool = False
    measurements: Measurements = Measurements()

    network: Network = Network()

    probabilistic: Probabilistic = Probabilistic()

    regression: Regression = Regression()


default_conf = OmegaConf.structured(Config)


def parse_args(argv=None):
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter

    description = 'A2S and S2A regressor'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfgs',
                        nargs='+', default=[],
                        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*',
                        help='The configuration of the Detector')
    cmd_args = parser.parse_args()

    cfg = default_conf.copy()
    for exp_cfg in cmd_args.exp_cfgs:
        if exp_cfg:
            cfg.merge_with(OmegaConf.load(exp_cfg))
    if cmd_args.exp_opts:
        cfg.merge_with(OmegaConf.from_cli(cmd_args.exp_opts))

    return cfg


def get_features_from_config(cfg):
    """ Select the attributes and measurements that are True in config file."""

    # attribute names are different in the synthetic dataset and can not
    # be selected
    if 'synthetic' not in cfg.get('dataset', ['caesar']):
        ds_gender = cfg.get('ds_gender', 'female')
        attribute_names = ATTRIBUTE_NAMES[ds_gender]

        # select the attributes that are used as input features
        attributes = []
        if cfg.get('use_attributes', True):
            attributes_conf = cfg.get(f'{ds_gender}_attributes')
            logger.info(attributes_conf)
            attributes = [k for k, v in attributes_conf.items() if v]

        # get index of attribute in vector. Verify that all selected
        # attributes are present in the constant list.
        attributes_idx = []
        if len(attributes) > 0:
            attributes_idx = np.array(
                [i for i, v in enumerate(attribute_names)
                 if v.lower().replace(' ', '_') in attributes])
        assert len(attributes_idx) == len(attributes), \
            'Some selected attributes are not annotated.'

    else:
        attributes = ATTRIBUTE_NAMES_SYNTHETIC_DATA
        attributes_idx = np.arange(len(attributes))
        logger.warning(f'Synthetic dataset used: all attributes will be used!')

    # select the measurements that are used as input features
    mmts = []
    if cfg.get('use_measurements', True):
        mmts_conf = cfg.get('measurements', {})
        mmts = [k for k, v in mmts_conf.items() if v]

    logger.info(f'{len(attributes)} selected attributes: {attributes}')
    logger.info(f'{len(mmts)} selected measurements: {mmts}')

    return attributes, attributes_idx, mmts
