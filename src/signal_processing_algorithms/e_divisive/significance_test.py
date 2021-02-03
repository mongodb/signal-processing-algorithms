"""Significance Tester for E-Divisive."""
import copy
import random

from typing import Iterable

import numpy as np

from more_itertools import pairwise

from signal_processing_algorithms.e_divisive.base import EDivisiveCalculator, SignificanceTester
from signal_processing_algorithms.e_divisive.change_points import EDivisiveChangePoint


class QHatPermutationsSignificanceTester(SignificanceTester):
    """A permutation tester for E-Divisive."""

    def __init__(self, calculator: EDivisiveCalculator, pvalue: float, permutations: int) -> None:
        """
        Create a default permutation tester.

        :param calculator: The calculator to use when calculating diffs an qhat values.
        """
        self._calculator = calculator
        self._pvalue = pvalue
        self._permutations = permutations

    def _permutation_test(self, series: np.ndarray, windows: Iterable[int]) -> float:
        """
        Perform a permutation test and return the highest q-hat value found.

        :param windows: The change point locations.
        :param series: The series to permute.
        :return: The maximum Q-Hat value.
        """
        permute_test_qhat_values = []
        for a, b in pairwise(windows):
            window = copy.copy(series[a:b])
            # Backwards compatibility with Python 2.7 implementation, see:
            # https://stackoverflow.com/questions/38943038/difference-between-python-2-and-3-for-shuffle-with-a-given-seed
            random.shuffle(window, random.random)
            permuted_diffs = self._calculator.calculate_diffs(window)
            permuted_qhat_values = self._calculator.calculate_qhat_values(permuted_diffs)
            permute_test_qhat_values.append(max(permuted_qhat_values))
        return max(permute_test_qhat_values)

    def is_significant(
        self, candidate: EDivisiveChangePoint, series: np.ndarray, windows: Iterable[int]
    ) -> bool:
        """
        Perform a significance test given the change point candidate, series and existing change point indices.

        :param candidate: The new change point candidate
        :param series: The series.
        :param windows: The change point indices defining windows.
        :return: Boolean value indicating whether the point is significant
        """
        # RANDOMLY PERMUTE CLUSTERS FOR SIGNIFICANCE TEST
        permutes_higher_qhat_count = 0
        for _ in range(self._permutations):
            best_permuted_qhat = self._permutation_test(series, windows)
            if best_permuted_qhat >= candidate.qhat:
                permutes_higher_qhat_count += 1
        candidate.probability = permutes_higher_qhat_count / (self._permutations + 1)
        return candidate.probability <= self._pvalue
