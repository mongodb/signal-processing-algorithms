"""Numpy implementation of E-Divisive."""
from typing import Iterable

import numpy as np

from typing_extensions import Protocol, runtime_checkable

from signal_processing_algorithms.e_divisive.change_points import EDivisiveChangePoint


@runtime_checkable
class EDivisiveCalculator(Protocol):
    """A Calculator for E-Divisive q-hat metrics and difference matrices."""

    @staticmethod
    def calculate_diffs(series: np.ndarray) -> np.ndarray:
        """
        Calculate the difference matrix of a series.

        :param series: The series.
        """
        ...

    @staticmethod
    def calculate_qhat_values(diffs: np.ndarray) -> np.ndarray:
        """
        Calculate the q-hat values of a difference matrix.

        :param diffs: The difference matrix.
        """
        ...


@runtime_checkable
class SignificanceTester(Protocol):
    """A permutation tester for E-Divisive."""

    def is_significant(
        self, candidate: EDivisiveChangePoint, series: np.ndarray, windows: Iterable[int]
    ) -> bool:
        """
        Perform a significance test given the change point candidate, series and existing change point indices.

        :param candidate: The new change point candidate
        :param series: The series.
        :param windows: The change point indices defining windows.
        :return: Boolean value indicating whether the point is significant.
        """
        ...
