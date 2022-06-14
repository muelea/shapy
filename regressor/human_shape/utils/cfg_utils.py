from loguru import logger
from yacs.config import CfgNode as CN
from omegaconf import OmegaConf

BUILTINS = [list, dict, tuple, set, str, int, float, bool]


def cfg_to_dict(cfg_node):
    if isinstance(cfg_node, (CN,)):
        return yacs_cfg_to_dict(cfg_node)
    elif OmegaConf.is_config(cfg_node):
        return OmegaConf.to_container(cfg_node)
    else:
        raise ValueError(f'Unknown object type: {type(cfg_node)}')


def yacs_cfg_to_dict(cfg_node):
    if type(cfg_node) in BUILTINS:
        return cfg_node
    else:
        curr_dict = dict(cfg_node)
        for key, val in curr_dict.items():
            curr_dict[key] = cfg_to_dict(val)
        return curr_dict
