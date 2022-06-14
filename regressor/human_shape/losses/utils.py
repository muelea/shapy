import torch

def get_reduction_method(reduction='mean'):
    if reduction == 'mean':
        return torch.mean
    elif reduction == 'sum':
        return torch.sum
    elif reduction == 'none':
        return lambda x: x
    else:
        raise ValueError(f'Unknown reduction method: {reduction}')
