"""Energy statistics."""
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from signal_processing_algorithms.distance import get_distance_matrix


@dataclass
class EnergyStatisticsTestResult:
    """Class for representing Energy Statistics and permutation test result."""

    e: float
    t: float
    h: float
    e_p: float
    t_p: float
    h_p: float


class EnergyStatistics:
    """
    Energy Statistics of distributions x and y.

    Calculates the following energy statistics of distributions x and y:
    1. E-statistic
    2. Test statistic
    3. E-coefficient of inhomogeneity
    """

    def __init__(
        self,
        x: Union[List[float], List[List[float]], np.ndarray],
        y: Union[List[float], List[List[float]], np.ndarray],
    ) -> None:
        """
        Initialize.

        :param x: A distribution which is an m x n array of m observations for n variables.
        :param y: A distribution which is an l x n array of l observations for n variables.
        """
        self._x = get_valid_input(x)
        self._y = get_valid_input(y)

        if self._x.shape[1] != self._y.shape[1]:
            raise ValueError(
                "Number of variables per observation must be the same for both distributions!"
            )

        distances_within_x = get_distance_matrix(self._x, self._x)
        distances_within_y = get_distance_matrix(self._y, self._y)
        distances_between_xy = get_distance_matrix(self._x, self._y)

        # E-statistic
        self.e = self.get_e_from_distance_matrices(
            distances_within_x, distances_within_y, distances_between_xy
        )

        self._n = len(self._x)  # Number of samples from X
        self._m = len(self._y)  # Number of samples from Y

        # Test statistic
        self._t_coefficient = (self._n * self._m) / (self._n + self._m)
        self.t = self.get_t_from_e(self._t_coefficient, self.e)

        # E-coefficient of inhomogeneity
        self.h = self.get_h_from_e_and_distance_matrix(self.e, distances_between_xy)

    def __repr__(self) -> str:
        """
        Energy statistics of distributions x and y.

        :return: String representation of energy statistics.
        """
        return f"EnergyStatistics(e:{self.e}, t:{self.t}, h:{self.h})"

    @staticmethod
    def get_e_from_distance_matrices(
        distances_within_x: np.ndarray,
        distances_within_y: np.ndarray,
        distances_between_xy: np.ndarray,
    ) -> float:
        """
        Return E-statistic from distance matrices.

        :param distances_within_x: Distance matrix of pairwise distances within distribution x.
        :param distances_within_y: Distance matrix of pairwise distances within distribution y.
        :param distances_between_xy: Distance matrix of pairwise distances between
        distribution x and y.
        :return: E-statistic.
        """
        return float(
            2 * np.mean(distances_between_xy)
            - np.mean(distances_within_x)
            - np.mean(distances_within_y)
        )

    @staticmethod
    def get_t_from_e(t_coefficient: float, e_statistic: float) -> float:
        """
        Return T-statistic from E-statistic and T-coefficient.

        :param t_coefficient: T-coefficient.
        :param e_statistic: E-statistic.
        :return:T-statistic.
        """
        return t_coefficient * e_statistic

    @staticmethod
    def get_h_from_e_and_distance_matrix(e: float, distances_between_xy: np.ndarray) -> float:
        """
        Return H (E-coefficient of inhomogeneity) from E-statistic and distance matrix.

        :param e: E-statistic.
        :param distances_between_xy: Distance matrix of pairwise distances between
        distribution x and y.
        :return:
        """
        avg_distances_between_xy = np.mean(distances_between_xy)
        return float(e / (2 * avg_distances_between_xy)) if avg_distances_between_xy > 0 else 0.0

    def _get_statistics_from_distance_matrices(
        self,
        distances_within_x: np.ndarray,
        distances_within_y: np.ndarray,
        distances_within_xy: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Return energy statistics.

        :param distances_within_x: Distance matrix of pairwise distances within distribution x.
        :param distances_within_y: Distance matrix of pairwise distances within distribution y.
        :param distances_within_xy: Distance matrix of pairwise distances
        between distribution x and y.
        :return: E-statistic, T-statistic and H(E-coefficient of inhomogeneity)
        """
        e = self.get_e_from_distance_matrices(
            distances_within_x, distances_within_y, distances_within_xy
        )
        t = self.get_t_from_e(self._t_coefficient, e)
        h = self.get_h_from_e_and_distance_matrix(e, distances_within_xy)
        return e, t, h

    def _get_statistics_from_combined_distance_matrix(
        self, distance_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Return energy statistics for distance matrix.

        :param distance_matrix: Distance matrix of pairwise distances within distribution formed by combining distributions x and y.
        :return: E-statistic, T-statistic and H(E-coefficient of inhomogeneity)
        """
        len_distance_matrix = len(distance_matrix)
        distances_within_x = distance_matrix[0 : self._n, 0 : self._n]
        distances_within_y = distance_matrix[
            self._n : len_distance_matrix, self._n : len_distance_matrix
        ]
        distances_between_xy = distance_matrix[self._n : len_distance_matrix, 0 : self._n]
        return self._get_statistics_from_distance_matrices(
            distances_within_x, distances_within_y, distances_between_xy
        )

    def get_statistics_and_probabilities(self, num_tests: int = 1000) -> EnergyStatisticsTestResult:
        """
        Return energy statistics and the corresponding permutation test results.

        :param num_tests: Number of tests to be run.
        :return: Energy statistics and permutation test results.
        """
        combined = np.concatenate((self._x, self._y))
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
        for _ in range(num_tests):
            np.random.shuffle(row_indices)
            shuffled_distances = distances_between_all[np.ix_(row_indices, row_indices)]
            shuffled_e, shuffled_t, shuffled_h = self._get_statistics_from_combined_distance_matrix(
                shuffled_distances
            )
            if shuffled_e >= self.e:
                count_e += 1
            if shuffled_t >= self.t:
                count_t += 1
            if shuffled_h >= self.h:
                count_h += 1

        return EnergyStatisticsTestResult(
            **{
                "e": self.e,
                "e_p": count_e / num_tests,
                "t": self.t,
                "t_p": count_t / num_tests,
                "h": self.h,
                "h_p": count_h / num_tests,
            }
        )


def get_valid_input(series: Union[List[float], List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Return valid form of input.

    :param series: A distribution.
    :return: Return a valid input.
    """
    if series is None or len(series) == 0:
        raise ValueError("Distribution cannot be empty.")

    if not isinstance(series, np.ndarray):
        series = np.asarray(series, dtype=np.float64)

    if series.dtype is not np.float64:
        series = np.array(series, dtype=np.float64)

    if len(series.shape) == 1:
        series = np.atleast_2d(series).T

    return series
