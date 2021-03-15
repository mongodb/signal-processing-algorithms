"""Energy statistics."""
from dataclasses import dataclass
from typing import List, Union

import numpy as np

from signal_processing_algorithms.distance import get_distance_matrix
from signal_processing_algorithms.utils import convert


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
    x = get_valid_input(x)
    y = get_valid_input(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "Number of variables per observation must be the same for both distributions!"
        )

    distances_within_x = get_distance_matrix(x, x)
    distances_within_y = get_distance_matrix(y, y)
    distances_between_xy = get_distance_matrix(x, y)
    double_avg_distances_between_xy = 2 * np.mean(distances_between_xy)

    # E-statistic
    e = double_avg_distances_between_xy - np.mean(distances_within_x) - np.mean(distances_within_y)

    n = len(x)  # Number of samples from X
    m = len(y)  # Number of samples from Y

    # Test statistic
    t = (n * m / (n + m)) * e

    # E-coefficient of inhomogeneity
    h = (e / double_avg_distances_between_xy) if double_avg_distances_between_xy > 0 else 0.0
    return EnergyStatistics(e=float(e), t=float(t), h=float(h))  # type: ignore


def get_energy_statistics_from_distance_matrix(
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

    double_avg_distances_between_xy = 2 * np.mean(distances_between_xy)
    # E-statistic
    e = double_avg_distances_between_xy - np.mean(distances_within_x) - np.mean(distances_within_y)

    # Test statistic
    t = (n * m / (n + m)) * e

    # E-coefficient of inhomogeneity
    h = (e / double_avg_distances_between_xy) if double_avg_distances_between_xy > 0 else 0.0

    return EnergyStatistics(e=float(e), t=float(t), h=float(h))  # type: ignore


def get_valid_input(series: Union[List[float], List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Return valid form of input.

    :param series: A distribution.
    :return: Return a valid input.
    """
    if series is None or len(series) == 0:
        raise ValueError("Distribution cannot be empty.")

    return convert(series)
