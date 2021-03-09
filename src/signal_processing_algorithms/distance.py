"""Distance matrices."""
import numpy as np


def get_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return the matrix of pairwise distances between x and y.

    :param x: An m x n array of m observations for n variables.
    :param y: An l x n array of l observations for n variables.
    :return: An m x l array where (i,j)th value is the distance between the observation
    at i-th row of x and j-th row of y.
    """
    return np.linalg.norm(x[:, np.newaxis] - y, axis=2)
