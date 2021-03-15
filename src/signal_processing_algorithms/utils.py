"""Utils."""
from typing import List, Union

import numpy as np


def convert(series: Union[List[float], List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Convert series into a 2 Dimensional numpy array of floats.

    :param series: series.
    :return: 2 Dimensional numpy array.
    """
    if not isinstance(series, np.ndarray):
        series = np.asarray(series, dtype=np.float64)

    if series.dtype is not np.float64:
        series = np.array(series, dtype=np.float64)

    if len(series.shape) == 1:
        series = np.atleast_2d(series).T

    return series
