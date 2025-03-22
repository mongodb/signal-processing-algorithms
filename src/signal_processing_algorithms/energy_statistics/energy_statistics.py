"""Energy statistics."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from more_itertools import pairwise

from signal_processing_algorithms.energy_statistics.cext_calculator import (
    C_EXTENSION_LOADED,
    calculate_distance_matrix,
    calculate_largest_q,
    calculate_t_values,
)


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

    if not isinstance(series, np.ndarray):
        raise Exception("Series is not the expected type, np.ndarray.")

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


def _get_distance_matrix(series: np.ndarray, use_c_if_possible: bool = True) -> np.ndarray:
    """
    Return the matrix of pairwise distances within the series.

    :param series: An m x n array of m observations for n variables.
    :return: An m x m array where (i,j)th value is the distance between the observation
    at i-th row of series and j-th row of series.
    """
    if use_c_if_possible and C_EXTENSION_LOADED and series.shape[1] == 1:
        return calculate_distance_matrix(series.flatten())
    else:
        return np.linalg.norm(series[:, np.newaxis] - series, axis=2)


def _calculate_stats(x: float, y: float, xy: float, n: int, m: int) -> Tuple[float, float, float]:
    """
    Calculate the stats from the terms and coefficients.

    :param xy: The sum of distances between sample distributions X, Y.
    :param x: The sum of distances within sample distribution X.
    :param y: The sum of distances within sample distribution Y.
    :param n: The length of sample distribution X.
    :param m: The length of sample distribution Y.

    :return: The q value generated from the terms.
    """
    xy_avg = xy / (n * m) if n > 0 and m > 0 else 0
    x_avg = x / (n**2) if n > 0 else 0
    y_avg = y / (m**2) if m > 0 else 0

    # E-statistic
    e = 2 * xy_avg - x_avg - y_avg
    # Test statistic
    t = (n * m / (n + m)) * e
    # E-coefficient of inhomogeneity
    h = (e / (2 * xy_avg)) if xy_avg > 0 else 0.0
    return e, t, h


def _calculate_t_stats(distance_matrix: np.ndarray, use_c_if_possible: bool = True) -> np.ndarray:
    """
    Find t-statistic values given a distance matrix .

    :param distance_matrix: The distance matrix.
    :return: The t-statistic values.
    """
    if use_c_if_possible and C_EXTENSION_LOADED:
        return calculate_t_values(distance_matrix)

    statistics = np.zeros(len(distance_matrix), dtype=np.float64)

    # We will partition our signal into:
    # X = {Xi; 0 <= i < tau}
    # Y = {Yj; tau <= j < len(signal) }
    # and look for argmax(tau)Q(tau)

    # sum |Xi - Yj| for i < tau <= j
    xy = 0
    # sum |Xi - Xj| for i < j < tau
    x = 0
    # sum |Yi - Yj| for tau <= i < j
    y = 0

    for row in range(0, len(distance_matrix)):
        y += np.sum(distance_matrix[row, row:])

    for tau in range(0, len(distance_matrix)):
        # Since the distance matrix is symmetric,
        # we can consider the upper or lower diagonal triangle and double the sum for
        # calculating the pairwise distances within x as well as y.
        statistics[tau] = _calculate_stats(2 * x, 2 * y, xy, tau, len(distance_matrix) - tau)[1]
        column_delta = np.sum(distance_matrix[:tau, tau])
        row_delta = np.sum(distance_matrix[tau, tau:])

        xy = xy - column_delta + row_delta
        x = x + column_delta
        y = y - row_delta

    return statistics


def _calculate_largest_q(
    distance_matrix: np.ndarray, min_cluster_size: int, use_c_if_possible: bool = True
) -> Tuple[float, Optional[int]]:
    """
    Find the largest Q value using the E-Divisive Mean algorithm.

    :param distance_matrix: The distance matrix.
    :param min_cluster_size: The minimum number of data points for a cluster.
    :return: The value and the index of the maximum Q-value.
    """
    if min_cluster_size < 2:
        print("Invalid min cluster size!")
        exit(1)

    tmp_best: Tuple[float, Optional[int]] = (-1 * np.inf, None)
    if len(distance_matrix) < 2 * min_cluster_size:
        return tmp_best

    if use_c_if_possible and C_EXTENSION_LOADED:
        return calculate_largest_q(distance_matrix, min_cluster_size)

    def q(end_x: int, end_y: int, cache: dict) -> float:
        """
        Calculate the Q value for two given clusters.

        :param endx: Index from the end of the left cluster (X).
        :param end_y: Index from the end of the right cluster (Y).
        :param cache: By proceeding incrementally when calculating the largest Q value in a
          time series, intermediate values can be reused. This cache serves as such a storage.
        :return: Returns the calculated Q value.
        """
        # For more efficient computation, we compute the distance matrix of all values.
        # - We compute the absolute value (as the absolute value is used in the forumlas)
        # - The diagonal is '0'
        # - The matrix is symmetric
        # Data: [1, 1, 1, 2, 4, 7, 9]
        # distance_matrix (the matrix if formatted as it due to better visualization in the next
        # steps):
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 1 ,  1 ,  1 ,  0 ,  2 ,  5 ,  7 ]
        # [ 3 ,  3 ,  3 ,  2 ,  0 ,  3 ,  5 ]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]
        #
        # Example:
        # IMPORTANT NOTE: The index of this exxample starts at 1 and not at 0! To make the
        # connection to the formula clearer if you look at the example next to the paper.
        # For efficiency, we do not use the diagonal as these values are always 0.
        #
        # end_x = 3; end_y = 6
        # X = [1, 1, 1]; Y = [2, 4, 7]
        # -------------
        # How to calculate x, y, xy. Values marked with paranthesis will be summed up:
        #
        # Calculate X (SUM 1<=i<k<=len(X): (|Xi - Xk|))
        # [ 0 , (0), (0),  1 ,  3 ,  6 ,  8 ] # i = 1
        # [ 0 ,  0 , (0),  1 ,  3 ,  6 ,  8 ] # i = 2
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 1 ,  1 ,  1 ,  0 ,  2 ,  5 ,  7 ]
        # [ 3 ,  3 ,  3 ,  2 ,  0 ,  3 ,  5 ]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]
        # x = 0

        def x_original() -> float:
            # end_x = 2; end_y = 5 (indexed at 0)
            # range: [0, 1]
            # distance_matrix[0, 1:3]
            # distance_matrix[1, 2:3]
            return sum(np.sum(distance_matrix[row, row + 1 : end_x + 1]) for row in range(0, end_x))

        # Calculate Y (SUM 1<=j<k<=len(Y): (|Yj - Yk|))
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 1 ,  1 ,  1 ,  0 , (2), (5),  7 ] # j = 1
        # [ 3 ,  3 ,  3 ,  2 ,  0 , (3),  5 ] # j = 2
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]
        # y = 10

        def y_original() -> float:
            # end_x = 2; end_y = 5 (indexed at 0)
            # range: [3, 4]
            # distance_matrix[3, 4:6]
            # distance_matrix[4, 5:6]
            return sum(
                np.sum(distance_matrix[row, row + 1 : end_y + 1]) for row in range(end_x + 1, end_y)
            )

        # Calculate XY (SUM 1<=i<=len(X); 1<=j<=len(Y): (|Xi - Yj|))
        # [ 0 ,  0 ,  0 , (1), (3), (6),  8 ] # i = 1
        # [ 0 ,  0 ,  0 , (1), (3), (6),  8 ] # i = 2
        # [ 0 ,  0 ,  0 , (1), (3), (6),  8 ] # i = 3
        # [ 1 ,  1 ,  1 ,  0 ,  2 ,  5 ,  7 ]
        # [ 3 ,  3 ,  3 ,  2 ,  0 ,  3 ,  5 ]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]
        # xy = 30

        def xy_original() -> float:
            # end_x = 2; end_y = 5 (indexed at 0)
            # range: [0, 1, 2]
            # distance_matrix[0, 3:6]
            # distance_matrix[1, 3:6]
            # distance_matrix[2, 3:6]
            return sum(
                np.sum(distance_matrix[row, end_x + 1 : end_y + 1]) for row in range(0, end_x + 1)
            )

        # How is caching used?
        # Caching will be explained based on the example given above. We store in each step for the
        # n and m value the respective x, y and xy values. We re-use these values and just
        # add/substract the new values. For all examples, we consider the scenario of increasing n
        # and increasing m.
        #
        # ### Caching for 'x' - increase 'end_x' ###
        # Current: end_x = 3; end_y = 6 (marked with '()')
        # Next: end_x = 4; end_y = 6 (new values marked with '[]')
        # X = [1, 1, 1, 2]; Y = [4, 7] (X changed, Y changed)
        # New values: rows '1 until (end_x-1)'; column 'end_x'
        # [ 0 , (0), (0), [1],  3 ,  6 ,  8 ]
        # [ 0 ,  0 , (0), [1] , 3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 , [1],  3 ,  6 ,  8 ]
        # [ 1 ,  1 ,  1 ,  0 ,  2 ,  5 ,  7 ]
        # [ 3 ,  3 ,  3 ,  2 ,  0 ,  3 ,  5 ]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]
        #
        # ### Caching for 'x' - increase 'end_y' ###
        # Current: end_x = 3; end_y = 6 (marked with '()')
        # Next: end_x = 3; end_y = 7
        # X = [1, 1, 1]; Y = [2, 4, 7, 9] (Y changed)
        # New values: -
        # [ 0 , (0), (0),  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 , (0),  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 1 ,  1 ,  1 ,  0 ,  2 ,  5 ,  7 ]
        # [ 3 ,  3 ,  3 ,  2 ,  0 ,  3 ,  5 ]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]

        def x_cache() -> Optional[float]:
            # end_x = 3; end_y = 5 (indexed at 0)
            # distance_matrix[0:3, 3]
            prev_x = cache["x"].get(end_x - 1)
            if prev_x is not None:
                return prev_x + np.sum(distance_matrix[0:end_x, end_x])
            return None

        # ### Caching for 'y' - increase 'end_x' ###
        # Current: end_x = 3; end_y = 6 (marked with '()')
        # Next: end_x = 4; end_y = 6 (removed values marked with '<>')
        # X = [1, 1, 1, 2]; Y = [4, 7] (X changed, Y changed)
        # Removed values: row 'end_x'; columns '(end_x+1) until end_y'
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 1 ,  1 ,  1 ,  0 , <2>, <5>,  7 ]
        # [ 3 ,  3 ,  3 ,  2 ,  0 , (3),  5 ]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        #
        # ### Caching for 'y' - increase 'end_y' ###
        # Current: end_x = 3; end_y = 6 (marked with '()')
        # Next: end_x = 3; end_y = 7 (new values marked with '[]')
        # X = [1, 1, 1]; Y = [2, 4, 7, 9] (Y changed)
        # New values: rows '(end_x+1) until (end_y-1)'; column 'end_y'
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 0 ,  0 ,  0 ,  1 ,  3 ,  6 ,  8 ]
        # [ 1 ,  1 ,  1 ,  0 , (2), (5), [7]]
        # [ 3 ,  3 ,  3 ,  2 ,  0 , (3), [5]]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 , [2]]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]

        def y_cache() -> Optional[float]:
            prev_y = cache["y"].get((end_x - 1, end_y))
            if prev_y is not None:
                # end_x = 3; end_y = 5 (indexed at 0)
                # distance_matrix[3, 4:6]
                return prev_y - np.sum(distance_matrix[end_x, end_x + 1 : end_y + 1])

            prev_y = cache["y"].get((end_x, end_y - 1))
            if prev_y is not None:
                # end_x = 2; end_y = 6 (indexed at 0)
                # distance_matrix[3:6, 6]
                return prev_y + np.sum(distance_matrix[end_x + 1 : end_y, end_y])

            return None

        # ### Caching for 'xy' - increase 'end_x' ###
        # Current: end_x = 3; end_y = 6 (marked with '()')
        # Next: end_x = 4; end_y = 6 (new values marked with '[]'; removed values marked with '<>')
        # X = [1, 1, 1, 2]; Y = [4, 7] (X changed, Y changed)
        # New values: row 'end_x'; columns '(end_x+1) until end_xy'
        # Removed values: rows '1 until (end_x-1)'; column 'end_x'
        # [ 0 ,  0 ,  0 , <1>, (3), (6),  8 ]
        # [ 0 ,  0 ,  0 , <1>, (3), (6),  8 ]
        # [ 0 ,  0 ,  0 , <1>, (3), (6),  8 ]
        # [ 1 ,  1 ,  1 ,  0 , [2], [5],  7 ]
        # [ 3 ,  3 ,  3 ,  2 ,  0 ,  3 ,  5 ]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]
        #
        # ### Caching for 'xy' - increase 'm' ###
        # Current: end_x = 3; end_y = 6 (marked with '()')
        # Next: end_x = 3; end_y = 7 (new values marked with '[]')
        # X = [1, 1, 1]; Y = [2, 4, 7, 9] (Y changed)
        # New values: rows '1 until end_x'; column 'end_y'
        # [ 0 ,  0 ,  0 , (1), (3), (6), [8]]
        # [ 0 ,  0 ,  0 , (1), (3), (6), [8]]
        # [ 0 ,  0 ,  0 , (1), (3), (6), [8]]
        # [ 1 ,  1 ,  1 ,  0 ,  2 ,  5 ,  7 ]
        # [ 3 ,  3 ,  3 ,  2 ,  0 ,  3 ,  5 ]
        # [ 6 ,  6 ,  6 ,  5 ,  3 ,  0 ,  2 ]
        # [ 8 ,  8 ,  8 ,  7 ,  5 ,  2 ,  0 ]

        def xy_cache() -> Optional[float]:
            prev_xy = cache["xy"].get((end_x - 1, end_y))
            if prev_xy is not None:
                # end_x = 3; end_y = 5 (indexed at 0)
                # distance_matrix[3, 4:6]
                # distance_matrix[0:3, 3]
                return (
                    prev_xy
                    + np.sum(distance_matrix[end_x, end_x + 1 : end_y + 1])
                    - np.sum(distance_matrix[0:end_x, end_x])
                )
            prev_xy = cache["xy"].get((end_x, end_y - 1))
            if prev_xy is not None:
                # end_x = 2; end_y = 6 (indexed at 0)
                # distance_matrix[0:3, 6]
                return prev_xy + np.sum(distance_matrix[0 : end_x + 1, end_y])
            return None

        x = x_cache()
        if x is None:
            x = x_original()

        y = y_cache()
        if y is None:
            y = y_original()

        xy = xy_cache()
        if xy is None:
            xy = xy_original()

        cache["x"][end_x] = x
        cache["y"][(end_x, end_y)] = y
        cache["xy"][(end_x, end_y)] = xy

        # Math: How to get rid of the binomial coefficient
        # n choose 2
        # = n! / (2! * (n-2)!)
        # = n * (n-1) * (n-2)! / (2 * (n-2)!)
        # = n * (n-1) / 2

        x_size = end_x + 1
        y_size = (end_y + 1) - x_size
        xy = (2.0 / (x_size * y_size)) * xy
        x = (2.0 / (x_size * (x_size - 1))) * x
        y = (2.0 / (y_size * (y_size - 1))) * y
        e = xy - x - y
        return ((x_size * y_size) / (x_size + y_size)) * e

    tmp_cache: Dict[str, Dict[int, float]] = {"x": {}, "y": {}, "xy": {}}
    for tau in range(min_cluster_size - 1, len(distance_matrix) - min_cluster_size):
        for kappa in range(tau + min_cluster_size, len(distance_matrix)):
            # Create two clusters out of the time series
            # 1 <= tau < kappa <= len(distance_matrix:
            # Xt = [Z1, Z1, ..., Zt]
            # Yk = [Zt+1, Zt+1, ... Zk]
            tmp = q(tau, kappa, tmp_cache)
            if tmp > tmp_best[0]:
                tmp_best = (tmp, tau + 1)

    return tmp_best


def _get_next_significant_change_point(
    distances: np.ndarray,
    change_points: List[int],
    memo: Dict[Tuple[int, int], Tuple[int, float]],
    pvalue: float,
    permutations: int,
    min_cluster_size: Optional[int],
) -> Optional[int]:
    """
    Calculate the next significant change point.

    Return the next significant change point within windows bounded by pre-existing change points
     (if there are any).

    :param distances: Distance matrix of pairwise distances between the values in the series.
    :param change_points: change points found until now.
    :param memo: cache.
    :param pvalue: p value for the permutation test.
    :param permutations: Number of permutations for the permutation test.
    :param min_cluster_size: The minimum number of data points for a cluster.
    If min_cluster_size is 'None', the old (original) algorithm is used.
    If min_cluster_size is >= 2, the E-DIvisive Mean algorithm is used.
    :return: The next most likely change point if one exists.
    """
    windows = [0] + sorted(change_points) + [len(distances)]
    candidates: List[Tuple[int, float]] = []
    for bounds in pairwise(windows):
        if bounds in memo:
            candidates.append(memo[bounds])
        else:
            a, b = bounds
            if min_cluster_size is None:
                stats = _calculate_t_stats(distances[a:b, a:b])
                idx = int(np.argmax(stats))
                largest_q = stats[idx]
            else:
                largest_q, idx = _calculate_largest_q(  # type:ignore[assignment]
                    distances[a:b, a:b], min_cluster_size
                )
                if idx is None:
                    # Cluster too small
                    continue
            new = (idx + a, largest_q)
            candidates.append(new)
            memo[bounds] = new
    if len(candidates) == 0:
        return None
    best_candidate = max(candidates, key=lambda x: x[1])
    better_num = 0
    for _ in range(permutations):
        permute_t = []
        for a, b in pairwise(windows):
            row_indices = np.arange(b - a) + a
            np.random.shuffle(row_indices)
            shuffled_distances = distances[np.ix_(row_indices, row_indices)]
            if min_cluster_size is None:
                stats = _calculate_t_stats(shuffled_distances)
                largest_q = max(stats)
            else:
                largest_q, idx = _calculate_largest_q(  # type:ignore[assignment]
                    shuffled_distances, min_cluster_size
                )
                if idx is None:
                    # Cluster too small
                    continue
            permute_t.append(largest_q)
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
    e, t, h = _calculate_stats(x, y, xy, n, m)
    return EnergyStatistics(e=e, t=t, h=h)


def e_divisive(
    series: Union[List[float], List[List[float]], np.ndarray],
    pvalue: float = 0.05,
    permutations: int = 100,
    min_cluster_size: Optional[int] = None,
) -> List[int]:
    """
    Calculate the change points in the series using e divisive.

    :param series: A series.
    :param pvalue: p value for the permutation test.
    :param permutations: Number of permutations for the permutation test.
    :param min_cluster_size: The minimum number of data points for a cluster.
    If min_cluster_size is 'None', the old (original) algorithm is used.
    If min_cluster_size is >= 2, the E-DIvisive Mean algorithm is used.
    :return: The indices of change points.
    """
    series = _get_valid_input(series)
    change_points: List[int] = []
    distances = _get_distance_matrix(series)
    memo: Dict[Tuple[int, int], Tuple[int, float]] = {}
    while significant_change_point := _get_next_significant_change_point(
        distances, change_points, memo, pvalue, permutations, min_cluster_size
    ):
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
    total = (
        permutations + 1
    )  # the first observation will be one of the permutations in the set of all permutations.
    return EnergyStatisticsWithProbabilities(
        e=energy_statistics.e,
        e_pvalue=count_e / total,
        t=energy_statistics.t,
        t_pvalue=count_t / total,
        h=energy_statistics.h,
        h_pvalue=count_h / total,
    )
