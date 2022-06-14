from typing import Union, Optional
import os
import os.path as osp
from omegaconf import DictConfig

from loguru import logger
from .body_models import SMPL, SMPLH, SMPLX


def build_body_model(body_model_cfg: DictConfig) -> Union[SMPL, SMPLH, SMPLX]:
    model_type = body_model_cfg.get('type', 'smplx')
    model_folder = osp.expandvars(
        body_model_cfg.get('model_folder', 'data/models'))

    curr_model_cfg = body_model_cfg.get(model_type, {})
    #  logger.info(f'{model_type}: {curr_model_cfg.pretty()}')

    logger.debug(f'Building {model_type.upper()} body model')
    model_path = osp.join(model_folder, model_type)
    if model_type.lower() == 'smpl':
        model = SMPL(model_path, **curr_model_cfg)
    elif model_type.lower() == 'smplh':
        model = SMPLH(model_path, **curr_model_cfg)
    elif model_type.lower() == 'smplx':
        model = SMPLX(model_path, **curr_model_cfg)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')
    return model
