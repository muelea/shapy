from loguru import logger
from .registry import BODY_HEAD_REGISTRY


def build(exp_cfg):
    network_cfg = exp_cfg.get('network', {})
    body_cfg = exp_cfg.get('body_model', {})

    network_type = network_cfg.get('type', 'smplx')
    if network_type == 'SMPLRegressor':
        network_cfg = network_cfg.get('smpl', {})
        loss_cfg = exp_cfg.get('losses', {}).get('body', {})
    elif network_type == 'SMPLHRegressor':
        network_cfg = network_cfg.get('smplh', {})
        loss_cfg = exp_cfg.get('losses', {}).get('body', {})
    elif network_type == 'SMPLXRegressor':
        network_cfg = network_cfg.get('smplx', {})
        loss_cfg = exp_cfg.get('losses', {}).get('body', {})
    elif network_type == 'SMPLGroupRegressor':
        network_cfg = network_cfg.get('smpl', {})
        loss_cfg = exp_cfg.get('losses', {}).get('body', {})
    elif network_type == 'SMPLHGroupRegressor':
        network_cfg = network_cfg.get('smplh', {})
        loss_cfg = exp_cfg.get('losses', {}).get('body', {})
    elif network_type == 'SMPLXGroupRegressor':
        network_cfg = network_cfg.get('smplx', {})
        loss_cfg = exp_cfg.get('losses', {}).get('body', {})
    else:
        raise ValueError(f'Unknown network type: {network_type}')

    return BODY_HEAD_REGISTRY.get(network_type)(
        body_cfg, network_cfg=network_cfg, loss_cfg=loss_cfg)
