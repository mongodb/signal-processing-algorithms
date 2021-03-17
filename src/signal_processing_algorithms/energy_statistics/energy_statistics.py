"""Energy statistics."""
from dataclasses import dataclass
from typing import List, Union, Optional

import numpy as np
from more_itertools import pairwise
from signal_processing_algorithms.energy_statistics.cext_calculator import C_EXTENSION_LOADED, calculate_diffs, \
    calculate_t_values


def _get_distance_matrix(series: np.ndarray, use_c_if_possible=True) -> np.ndarray:
    """
    Return the matrix of pairwise distances between x and y.

    :param x: An m x n array of m observations for n variables.
    :param y: An l x n array of l observations for n variables.
    :return: An m x l array where (i,j)th value is the distance between the observation
    at i-th row of x and j-th row of y.
    """
    if use_c_if_possible and C_EXTENSION_LOADED and series.shape[1] == 1:
        return calculate_diffs(series.flatten())
    else:
        return np.linalg.norm(series[:, np.newaxis] - series, axis=2)


@dataclass
class EnergyStatistics:
    """
    Class for representing Energy Statistics.

    e - E-statistic
    t - Test statistic
    h - E-coefficient of inhomogeneity
    """

    e: float
    t: float
    h: float


@dataclass
class EnergyStatisticsWithProbabilities(EnergyStatistics):
    """Class for representing Energy Statistics and permutation test result."""

    e_pvalue: float
    t_pvalue: float
    h_pvalue: float


def _calculate_stats(x: float, y: float, xy: float, n: int, m: int) -> (float, float, float):
    """
    Calculate the q value from the terms and coefficients.

    :param xy: The sum of differences between sample distributions X, Y.
    :param x: The sum of differences within sample distribution X.
    :param y: The sum of differences within sample distribution Y.
    :param n: The length of sample distribution X.
    :param m: The length of sample distribution Y.

    :return: The q value generated from the terms.
    """

    xy_avg = xy / (n*m) if n > 0 and m > 0 else 0
    x_avg = x / (n**2) if n > 0 else 0
    y_avg = y / (m**2) if m > 0 else 0

    # E-statistic
    e = 2 * xy_avg - x_avg - y_avg
    # Test statistic
    t = (n * m / (n + m)) * e
    # E-coefficient of inhomogeneity
    h = (e / (2 * xy_avg)) if xy_avg > 0 else 0.0
    return e, t, h


def _calculate_t_stats(diffs: np.ndarray, use_c_if_possible=True) -> np.ndarray:
    """
    Find e-statistics values given a difference matrix.

    :param diffs: The difference matrix.
    :return: The qhat values.
    """
    if use_c_if_possible and C_EXTENSION_LOADED:
        return calculate_t_values(diffs)

    statistics = np.zeros(len(diffs), dtype=np.float64)

    # We will partition our signal into:
    # X = {Xi; 0 <= i < tau}
    # Y = {Yj; tau <= j < len(signal) }
    # and look for argmax(tau)Q(tau)

    # sum |Xi - Yj| for i < tau <= j
    xy = 0
    # sum |Xi - Xj| for i < j < tau
    x = 0
    # sum |Yi - Yj| for tau <= i < j
    y = np.sum(diffs)

    for tau in range(0, len(diffs)):
        statistics[tau] = _calculate_stats(x, y, xy, tau, len(diffs) - tau)[1]
        left, up, right, down = (np.sum(piece) for piece in (diffs[tau, :tau], diffs[:tau, tau], diffs[tau, tau:], diffs[tau:, tau]))
        x_delta = left + up
        y_delta = - right - down
        xy_delta = - y_delta - x_delta

        xy += xy_delta
        x += x_delta
        y += y_delta

    return statistics


def _get_next_significant_change_point(distances, change_points, memo, pvalue, permutations) -> Optional[int]:
    windows = [0] + change_points + [len(distances)]
    candidates = []
    for bounds in pairwise(windows):
        if bounds in memo:
            candidates.append(memo[bounds])
        else:
            a, b = bounds
            stats = _calculate_t_stats(distances[a:b, a:b])
            idx = np.argmax(stats)
            new = (idx + a, stats[idx])
            candidates.append(new)
            memo[bounds] = new
    best_candidate = max(candidates, key=lambda x: x[1])
    better_num = 0
    for _ in range(permutations):
        permute_t = []
        for a, b in pairwise(windows):
            row_indices = np.arange(b-a) + a
            np.random.shuffle(row_indices)
            shuffled_distances = distances[np.ix_(row_indices, row_indices)]
            stats = _calculate_t_stats(shuffled_distances)
            permute_t.append(max(stats))
        best = max(permute_t)
        if best >= best_candidate[1]:
            better_num += 1
    probability = better_num / (permutations + 1)
    return best_candidate[0] if probability <= pvalue else None


def _get_energy_statistics_from_distance_matrix(
    distance_matrix: np.ndarray, n: int, m: int
) -> EnergyStatistics:
    """
    Return energy statistics from distance matrix.

    :param distance_matrix: Distance matrix of pairwise distances within distribution formed by combining distributions x and y.
    :param n: Number of samples in x.
    :param m: Number of samples in y.
    :return: Energy Statistics.
    """
    len_distance_matrix = len(distance_matrix)
    distances_within_x = distance_matrix[0:n, 0:n]
    distances_within_y = distance_matrix[n:len_distance_matrix, n:len_distance_matrix]
    distances_between_xy = distance_matrix[n:len_distance_matrix, 0:n]
    x, y, xy = np.sum(distances_within_x), np.sum(distances_within_y), np.sum(distances_between_xy)
    e,t, h = _calculate_stats(x, y, xy, n, m)
    return EnergyStatistics(e=e, t=t, h=h)


def _convert(series: Union[List[float], List[List[float]], np.ndarray]) -> np.ndarray:
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


def _get_valid_input(series: Union[List[float], List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Return valid form of input.

    :param series: A distribution.
    :return: Return a valid input.
    """
    if series is None or len(series) == 0:
        raise ValueError("Distribution cannot be empty.")

    return _convert(series)


def e_divisive(series, pvalue=0.05, permutations=100) -> List[int]:
    series = _get_valid_input(series)
    change_points = []
    distances = _get_distance_matrix(series)
    memo = {}
    while significant_change_point := _get_next_significant_change_point(distances, change_points, memo, pvalue, permutations):
        change_points.append(significant_change_point)
    return change_points


def get_energy_statistics(
    x: Union[List[float], List[List[float]], np.ndarray],
    y: Union[List[float], List[List[float]], np.ndarray],
) -> EnergyStatistics:
    """
    Calculate energy statistics of distributions x and y.

    The following statistics are calculated:
    1. E-statistic
    2. Test statistic
    3. E-coefficient of inhomogeneity

    :param x: A distribution which is an m x n array of m observations for n variables.
    :param y: A distribution which is an l x n array of l observations for n variables.
    :return: Energy statistics of distributions x and y.
    """
    xy = np.concatenate((_get_valid_input(x), _get_valid_input(y)))
    distances = _get_distance_matrix(xy)
    return _get_energy_statistics_from_distance_matrix(distances, len(x), len(y))


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
    x = _get_valid_input(x)
    y = _get_valid_input(y)

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

    distances_between_all = _get_distance_matrix(combined)
    len_combined = len(combined)

    count_e = 0
    count_t = 0
    count_h = 0

    row_indices = np.arange(len_combined)

    n = len(x)
    m = len(y)

    energy_statistics = _get_energy_statistics_from_distance_matrix(distances_between_all, n, m)
    for _ in range(permutations):
        np.random.shuffle(row_indices)
        # np.ix_()'s main use is to create an open mesh so that we can use it to
        # select specific indices from an array (specific sub-array).
        # We use it here to rearrange the combined distance matrix.
        shuffled_distances = distances_between_all[np.ix_(row_indices, row_indices)]
        shuffled_energy_statistics = _get_energy_statistics_from_distance_matrix(
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
