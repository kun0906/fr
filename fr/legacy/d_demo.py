"""

https://towardsdatascience.com/intuitions-in-high-dimensional-spaces-c22f0441ce19
"""

from functools import partial

import numpy as np


def lp_distance(p_norm: int, point_one: np.ndarray, point_two: np.ndarray) -> float:
    """
    Calculate L sub p_norm norm for distance between point one and point two.
    Args:
        point_one: Point one.
        point_two: Point two.
    Returns:
        Lp norm distance.
    """

    return np.sum(np.abs(point_one - point_two) ** (p_norm)) ** (1 / p_norm)


l1_distance = partial(lp_distance, p_norm=1)
l2_distance = partial(lp_distance, p_norm=2)

norms = (("l1", l1_distance), ("l2", l2_distance))
rng = np.random.RandomState(seed=42)
two_dimensional_samples = rng.normal(loc=0, scale=10, size=(100, 2))
ten_dimensional_samples = rng.normal(loc=0, scale=10, size=(100, 10))
thousand_dimensional_samples = rng.normal(loc=0, scale=10, size=(100, 1000))


dimensionality_samples = (
    ("two", two_dimensional_samples),
    ("ten", ten_dimensional_samples),
    ("thousand", thousand_dimensional_samples),
)

for norm_name, norm_function in norms:
    for dim_name, dim_samples in dimensionality_samples:

        origin = np.zeros_like(dim_samples)
        distances = [
            norm_function(point_one=sample, point_two=origin) for sample in dim_samples
        ]

        ratio = np.max(distances) / np.min(distances)

        print(
            f"{norm_name} farthest-to-nearest ratio is {ratio} for {dim_name}-dimensional data"
        )
