"""Calculate difference matrices."""
import numpy as np


def pairwise_difference(x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """
    Return the pairwise distance between points of a 1D array or two 1D arrays.

    :param x: A 1D array of n observations.
    :param y: A 1D array of m observations or None.
    :return: An n x m array where (i,j) th observation is the distance
    between x[i] and y[j]. If y is None the (i,j)th observation is the
    distance between x[i] and x[j]
    """
    if y is None:
        y = x
    diff = np.abs(x[:, None] - y)
    return diff
