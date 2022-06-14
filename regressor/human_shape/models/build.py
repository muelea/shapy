from typing import Dict, Optional, Union
import sys

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from .discriminator import build_discriminator
from .body_heads import build_body_head, BODY_HEAD_REGISTRY
from human_shape.losses import build_adv_loss


def build_model(exp_cfg) -> Dict[str, nn.Module]:
    network_cfg = exp_cfg.get('network', {})
    net_type = network_cfg.get('type', 'expose')

    logger.info(f'Going to build a: {net_type}')
    if net_type in BODY_HEAD_REGISTRY:
        network = build_body_head(exp_cfg)
    else:
        raise ValueError(f'Unknown network type: {net_type}')

    discriminator, disc_loss = None, None

    use_adv_training = exp_cfg.use_adv_training
    if use_adv_training:
        raise NotImplementedError
        discriminator = build_discriminator(
            exp_cfg.network.discriminator)
        disc_loss = build_adv_loss(discriminator, exp_cfg.losses.discriminator)
    return {
        'network': network,
        'discriminator': discriminator,
        'discriminator_loss': disc_loss,
    }
