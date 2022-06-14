import numpy as np
import open3d as o3d
from .typing import Array

__all__ = [
    'rel_change',
    'binarize',
    'max_grad_change',
    'to_np',
    'np2o3d_pcl',
]


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def binarize(
    array: Array,
    thresh: float = -1,
    dtype: type = np.float32
) -> Array:
    if thresh > 0:
        return (array >= thresh).astype(dtype)
    else:
        return (array > 0).astype(dtype)


def max_grad_change(grad_arr):
    return grad_arr.abs().max()


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def np2o3d_pcl(x: np.ndarray) -> o3d.geometry.PointCloud:
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(x)

    return pcl
