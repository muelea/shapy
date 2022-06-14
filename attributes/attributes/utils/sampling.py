import numpy as np
from .typing import Array


def sample_in_sphere(d: int, N: int = 1, max_radius: float = 1.0) -> Array:
    # an array of d normally distributed random variables
    u = np.random.normal(0, 1, (N, d))
    norm = np.sum(u ** 2, axis=-1, keepdims=True) ** (0.5)
    r = np.random.rand(N, 1) ** (1.0 / d) * max_radius
    return r * u / norm
