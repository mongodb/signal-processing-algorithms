"""Homogeneity Calculator."""
from typing import Callable, List, Tuple, Union

import numpy as np

from signal_processing_algorithms import difference


class HomogeneityCalculator:
    """A calculator for determining the homogeneity of two distributions."""

    def __init__(self, avg: Callable[..., float] = np.mean) -> None:  # type: ignore
        """
        Initialize.

        :param avg: Function to calculate average.
        """
        self._avg = avg

    @classmethod
    def get_valid_type(cls, series: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Return a valid series type.

        :param series: Distribution.
        :return: A 1D numpy array of floats.
        """
        if series is None or len(series) == 0:
            raise ValueError("Distribution cannot be empty.")

        if not isinstance(series, np.ndarray):
            series = np.array(series, dtype=np.float64)

        if series.dtype is not np.float64:
            series = np.array(series, dtype=np.float64)

        return series

    def _get_energy_coefficient_of_inhomogeneity(
        self,
        pairwise_distance_in_x: np.ndarray,
        pairwise_distance_in_y: np.ndarray,
        pairwise_distance_in_x_y: np.ndarray,
    ) -> float:
        """
        Return Energy coefficient of inhomogeneity from distance matrices.

        :param pairwise_distance_in_x: Distance matrix for pairwise distance in the distribution x.
        :param pairwise_distance_in_y: Distance matrix for pairwise distance in the distribution y.
        :return: Energy coefficient of inhomogeneity.
        """
        expectation_x = self._avg(pairwise_distance_in_x)
        expectation_y = self._avg(pairwise_distance_in_y)
        expectation_x_y = self._avg(pairwise_distance_in_x_y)

        common_term = 2 * expectation_x_y
        inhomogeneity_coefficient = (common_term - expectation_x - expectation_y) / common_term
        assert 0 <= inhomogeneity_coefficient <= 1

        return inhomogeneity_coefficient

    def get_inhomogeneity(
        self, x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]
    ) -> float:
        """
        Return the Energy coefficient of homogeneity of fitted distributions x and y.

        :param x: A 1D array of observations.
        :param y: A 1D array of observations.
        :return: Energy coefficient of homogeneity.
        """
        x = self.get_valid_type(x)
        y = self.get_valid_type(y)

        pairwise_distance_in_x = difference.pairwise_difference(x)
        pairwise_distance_in_y = difference.pairwise_difference(y)
        pairwise_distance_in_x_y = difference.pairwise_difference(x, y)

        inhomogeneity_coefficient = self._get_energy_coefficient_of_inhomogeneity(
            pairwise_distance_in_x, pairwise_distance_in_y, pairwise_distance_in_x_y
        )

        return inhomogeneity_coefficient

    def get_homogeneity(
        self, x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]
    ) -> float:
        """
        Return the Energy coefficient of homogeneity of fitted distributions x and y.

        :return: Energy coefficient of homogeneity.
        """
        return 1 - self.get_inhomogeneity(x, y)
