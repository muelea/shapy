from typing import Tuple
import numpy as np
import torch

from human_shape.utils import Array


def targets_to_array_and_indices(
    targets,
    field_key: str,
    data_key: str,
) -> Tuple[Array, Array]:
    indices = np.array([ii for ii, t in enumerate(targets) if
                        t.has_field(field_key)], dtype=np.int)
    if len(indices) > 1:
        data_lst = []
        for ii, t in enumerate(targets):
            if t.has_field(field_key):
                data = getattr(t.get_field(field_key), data_key)
                if torch.is_tensor(data):
                    data = data.detach().cpu().numpy()
                data_lst.append(data)
        data_array = np.stack(data_lst)
        return data_array, indices
    else:
        return np.array([]), indices
