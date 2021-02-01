"""Computes the E-Divisive means change points."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import structlog

from more_itertools import pairwise

from signal_processing_algorithms.determinism import deterministic_random
from signal_processing_algorithms.e_divisive.base import EDivisiveCalculator, SignificanceTester
from signal_processing_algorithms.e_divisive.change_points import EDivisiveChangePoint

LOG = structlog.get_logger()


class EDivisive:
    """Class to compute the E-Divisive means change points."""

    def __init__(
        self,
        seed: Optional[int],
        calculator: EDivisiveCalculator,
        significance_tester: SignificanceTester,
    ) -> None:
        """
        Implement the E-Divisive algorithm in python.

        :param seed: Random number generation seed, for deterministic results.
        :param calculator: The calculator used for calculating q-hat values and difference matrices.
        :param significance_tester: The significance tester to confirm change points.
        """
        self._calculator = calculator
        self._significance_tester = significance_tester
        self._memoized_change_points: Dict[Tuple[int, int], EDivisiveChangePoint] = {}
        self._seed = seed
        self._change_points: List[EDivisiveChangePoint] = []
        self._sorted_change_points: List[EDivisiveChangePoint] = []
        self._calculated = False
        self._series: np.ndarray = np.ndarray(0, dtype=np.float)
        self._diffs: np.ndarray = np.ndarray((0, 0), dtype=np.float)

    @staticmethod
    def _find_change_point(qhat_values: np.ndarray) -> EDivisiveChangePoint:
        """
        Find the change point in a series of q-hat metrics.

        :param qhat_values: The q-hat metrics.
        :return: The change point.
        """
        if qhat_values.size:
            max_q_index = np.argmax(qhat_values)
            max_q = qhat_values[max_q_index]
        else:
            max_q = 0
            max_q_index = 0

        return EDivisiveChangePoint(index=max_q_index, qhat=max_q)

    def _add_change_point(self, change_point: EDivisiveChangePoint) -> None:
        self._change_points.append(change_point)
        self._sorted_change_points = sorted(self._change_points, key=lambda x: x.index)

    def _reset_change_points(self) -> None:
        self._change_points.clear()
        self._sorted_change_points.clear()
        self._memoized_change_points = {}
        self._calculated = False

    def _calculate_best_change_point(self) -> EDivisiveChangePoint:
        """
        Calculate a change point within windows bounded by pre-existing change points (if there are any).

        :return: The next most likely change point.
        """
        new_change_point_candidates = []
        for bounds in pairwise(self._windows()):
            if bounds in self._memoized_change_points:
                new_change_point_candidate = self._memoized_change_points[bounds]
                new_change_point_candidates.append(new_change_point_candidate)
            else:
                a, b = bounds
                candidate_qhat_values = self._calculator.calculate_qhat_values(
                    self._diffs[a:b, a:b]
                )
                new_change_point_candidate = self._find_change_point(candidate_qhat_values)
                # correct for window offset
                new_change_point_candidate.index += a
                new_change_point_candidates.append(new_change_point_candidate)
                self._memoized_change_points[bounds] = new_change_point_candidate
        return max(new_change_point_candidates, key=lambda x: x.qhat)

    def _windows(self) -> Iterable[int]:
        """Generate window bounds based on existing change points."""
        return [0] + [c.index for c in self._sorted_change_points] + [len(self._series)]

    def _compute_change_points(self) -> None:
        """
        Compute change points for a series of floats/integers.

        :return: The change points.
        """
        self._diffs = self._calculator.calculate_diffs(self._series)
        best_candidate = self._calculate_best_change_point()
        # Compute additional change points, so long as they are significant
        while self._significance_tester.is_significant(
            best_candidate, self._series, self._windows()
        ):
            self._add_change_point(best_candidate)
            best_candidate = self._calculate_best_change_point()
        self._calculated = True

    def fit(self, series: Union[List[float], np.ndarray]) -> None:
        """
        Fit the algorithm to a series.

        :param series: The series.
        """
        if series is None:
            series = []
        if not isinstance(series, np.ndarray):
            series = np.array(series, dtype=np.float)
        if series.dtype is not np.float:
            series = np.array(series, dtype=np.float)
        self._series = series
        self._reset_change_points()

    def predict(self) -> List[int]:
        """
        Predict the change points for the fitted series.

        :return: The change points.
        """
        if self._calculated:
            return [cp.index for cp in self._sorted_change_points]
        if self._seed is not None:
            with deterministic_random(self._seed), np.errstate(all="raise"):
                self._compute_change_points()
        else:
            self._compute_change_points()
        return [cp.index for cp in self._sorted_change_points]

    def fit_predict(self, series: Union[List[float], np.ndarray]) -> List[int]:
        """
        Fit the algorithm to a new series and predict its change points.

        :param series: The series to fit.
        :return: The change points.
        """
        self.fit(series)
        return self.predict()

    def get_change_points(
        self, series: Union[List[float], np.ndarray]
    ) -> List[EDivisiveChangePoint]:
        """
        Calculate change points for a series of floats.

        :param series: The series of floats.
        :return: The list of change points.
        """
        self.fit(series)
        self.predict()
        return self._change_points
