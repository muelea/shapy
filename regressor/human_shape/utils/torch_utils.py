from typing import Dict
import torch

from .typing import Tensor


def no_reduction(arg):
    return arg


def tensor_scalar_dict_to_float(tensor_dict: Dict[str, Tensor]):
    return {key: val.detach() if torch.is_tensor(val) else val
            for key, val in tensor_dict.items()}


def to_tensor(tensor, device=None, dtype=torch.float32):
    if isinstance(tensor, torch.Tensor):
        return tensor
    else:
        return torch.tensor(tensor, dtype=dtype, device=device)


def get_reduction_method(reduction='mean'):
    if reduction == 'mean':
        reduction = torch.mean
    elif reduction == 'sum':
        reduction = torch.sum
    elif reduction == 'none':
        reduction = no_reduction
    else:
        raise ValueError('Unknown reduction type: {}'.format(reduction))
    return reduction


def tensor_to_numpy(tensor, default=None):
    if tensor is None:
        return default
    else:
        return tensor.detach().cpu().numpy()
