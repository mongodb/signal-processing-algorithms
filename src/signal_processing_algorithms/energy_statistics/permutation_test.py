"""Permutation tests for Energy Statistics."""
from typing import List, Union

import numpy as np

from signal_processing_algorithms.distance import get_distance_matrix
from signal_processing_algorithms.energy_statistics.energy_statistics import (
    EnergyStatisticsWithProbabilities,
    get_energy_statistics,
    get_energy_statistics_from_distance_matrix,
    get_valid_input,
)


def get_energy_statistics_and_probabilities(
    x: Union[List[float], List[List[float]], np.ndarray],
    y: Union[List[float], List[List[float]], np.ndarray],
    permutations: int = 1000,
) -> EnergyStatisticsWithProbabilities:
    """
    Return energy statistics and the corresponding permutation test results.

    :param x: A distribution which is an m x n array of m observations for n variables.
    :param y: A distribution which is an l x n array of l observations for n variables.
    :param permutations: Number of tests to be run.
    :return: Energy statistics and permutation test results.
    """
    x = get_valid_input(x)
    y = get_valid_input(y)

    combined = np.concatenate((x, y))
    """
    Shuffling the combined matrices and finding distances can be optimized by 
    calculating the combined distance matrix and rearranging the distance matrix. 
    For example, 
    x = [[1, 2], [3, 4], [5, 6]]
    y = [[7, 8], [9, 10]]

    Combined matrix is:
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    ----------------------------------------------
    Pairwise distances within Combined matrix is:
    ----------------------------------------------
    [
        [ 0.          2.82842712  5.65685425  8.48528137 11.3137085 ]
        [ 2.82842712  0.          2.82842712  5.65685425  8.48528137]
        [ 5.65685425  2.82842712  0.          2.82842712  5.65685425]
        [ 8.48528137  5.65685425  2.82842712  0.          2.82842712]
        [11.3137085   8.48528137  5.65685425  2.82842712  0.        ]
    ]
    -----------------------------
    Shuffling the combined array:
    -----------------------------
    [
        [ 5.  6.]
        [ 9. 10.]
        [ 1.  2.]
        [ 3.  4.]
        [ 7.  8.]
    ]
    Splitting into x and y:
    -------
    x
    -------
    [
        [ 5.  6.]
        [ 9. 10.]
        [ 1.  2.]
    ]
    -------
    y
    -------
    [
        [3. 4.]
        [7. 8.]
    ]
    ----------------------------
    Pairwise distances within x:
    ----------------------------
    [
        [ 0.          5.65685425  5.65685425]
        [ 5.65685425  0.         11.3137085 ]
        [ 5.65685425 11.3137085   0.        ]
    ]
    ----------------------------
    Pairwise distances within y:
    ----------------------------
    [
        [0.         5.65685425]
        [5.65685425 0.        ]
    ]
    ------------------------------------
    Pairwise distances between x and y:
    ------------------------------------
    [
        [2.82842712 2.82842712]
        [8.48528137 2.82842712]
        [2.82842712 8.48528137]
    ]
    ------------------------------------
    This is same as the rearranging the combined distance matrix using a rearranged order [2 4 0 1 3] and 
    creating an open mesh using np.ix_([2 4 0 1 3], [2 4 0 1 3]) on combined matrix to get:
    [
        [ 0.          5.65685425  5.65685425  2.82842712  2.82842712]
        [ 5.65685425  0.         11.3137085   8.48528137  2.82842712]
        [ 5.65685425 11.3137085   0.          2.82842712  8.48528137]
        [ 2.82842712  8.48528137  2.82842712  0.          5.65685425]
        [ 2.82842712  2.82842712  8.48528137  5.65685425  0.        ]
    ]

    The sub matrix from top left corner (0,0) to bottom right corner (2,2) is the pairwise distances within x.
    The sub matrix from top left corner (3,3) to bottom right corner (4,4) is the pairwise distances within y.
    The sub matrix from top left corner (0,3) to bottom right corner (2,4) is the pairwise distances between 
    x and y.

    """

    distances_between_all = get_distance_matrix(combined, combined)
    len_combined = len(combined)

    count_e = 0
    count_t = 0
    count_h = 0

    row_indices = np.arange(len_combined)
    energy_statistics = get_energy_statistics(x, y)

    n = len(x)
    m = len(y)

    for _ in range(permutations):
        np.random.shuffle(row_indices)
        # np.ix_()'s main use is to create an open mesh so that we can use it to
        # select specific indices from an array (specific sub-array).
        # We use it here to rearrange the combined distance matrix.
        shuffled_distances = distances_between_all[np.ix_(row_indices, row_indices)]
        shuffled_energy_statistics = get_energy_statistics_from_distance_matrix(
            shuffled_distances, n, m
        )
        if shuffled_energy_statistics.e >= energy_statistics.e:
            count_e += 1
        if shuffled_energy_statistics.t >= energy_statistics.t:
            count_t += 1
        if shuffled_energy_statistics.h >= energy_statistics.h:
            count_h += 1

    return EnergyStatisticsWithProbabilities(
        e=energy_statistics.e,
        e_pvalue=count_e / permutations,
        t=energy_statistics.t,
        t_pvalue=count_t / permutations,
        h=energy_statistics.h,
        h_pvalue=count_h / permutations,
    )
