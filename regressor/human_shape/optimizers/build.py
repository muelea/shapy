from typing import Optional, List, Union
from omegaconf import DictConfig, OmegaConf

from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from human_shape.utils import TensorList


def build_optimizer(
        model: nn.Module,
        optim_cfg: DictConfig,
        exclude: str = ''
) -> Union[optim.SGD, optim.Adam, optim.RMSprop]:
    ''' Builds an optimizer object
    '''
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = optim_cfg.lr
        weight_decay = optim_cfg.weight_decay
        if "bias" in key:
            lr = optim_cfg.lr * optim_cfg.bias_lr_factor
            weight_decay = optim_cfg.weight_decay_bias

        if len(exclude) > 0 and exclude in key:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = get_optimizer(params, optim_cfg)
    return optimizer


def get_optimizer(
    params: TensorList,
    optim_cfg: DictConfig
) -> Union[optim.SGD, optim.Adam, optim.RMSprop]:
    lr = optim_cfg.lr
    optimizer_type = optim_cfg.type
    logger.info(f'Building optimizer: {optimizer_type.title()}')
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(
            params, lr, **OmegaConf.to_container(optim_cfg.sgd))
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            params, lr, **OmegaConf.to_container(optim_cfg.adam))
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            params, lr, **OmegaConf.to_container(optim_cfg.adam))
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            params, lr, **OmegaConf.to_container(optim_cfg.rmsprop))
    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_type}')
    return optimizer


def build_scheduler(optimizer, sched_cfg):
    scheduler_type = sched_cfg.type
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'step-lr':
        step_size = sched_cfg.step_size
        gamma = sched_cfg.gamma
        logger.info('Building scheduler: StepLR(step_size={}, gamma={})',
                    step_size, gamma)
        return scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler_type == 'multi-step-lr':
        gamma = sched_cfg.gamma
        milestones = sched_cfg.milestones
        logger.info('Building scheduler: MultiStepLR(milestone={}, gamma={})',
                    milestones, gamma)
        return scheduler.MultiStepLR(optimizer, milestones=milestones,
                                     gamma=gamma)
    else:
        raise ValueError('Unknown scheduler type: {}'.format(scheduler_type))
