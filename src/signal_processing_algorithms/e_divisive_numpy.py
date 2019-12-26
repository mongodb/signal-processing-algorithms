"""Numpy implementation of E-Divisive."""
import copy
import random
from contextlib import contextmanager
from typing import Union, List, NamedTuple, Optional, Generator

import numpy as np
import structlog

LOG = structlog.get_logger()


class QHatValues(NamedTuple):
    """A named tuple for the QHat values computed by the E-Divisive algorithm."""

    values: np.ndarray
    average_value: float
    average_diff: float
    length: int


class EDivisiveChangePoint(NamedTuple):
    """A named tuple for a change point returned by the E-Divisive algorithm."""

    index: int  # type: ignore
    value: float
    value_to_avg: float
    value_to_avg_diff: float
    average: float
    average_diff: float
    window_size: int
    probability: Optional[float]


# E-Divisive's definition requires it to permute change-windows
# which leads to non-determinism: we need to always get the
# same change-point results when running on the same input.
@contextmanager
def deterministic_random(seed: int) -> Generator:
    """
    Call random.seed(seed) during invocation and then restore state after.
    :param seed: RNG seed
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


class EDivisiveNumpyImplementation:
    """Class to compute the E-Divisive means change points."""

    def __init__(self, pvalue: float = 0.05, permutations: int = 100):
        """
        Implement the E-Divisive algorithm in python.

        :param pvalue: This the significance level for our testing.
        :param permutations: The max number of permutations to perform when evaluating the
        pvalue significance testing.
        """
        self.pvalue = pvalue
        self.permutations = permutations

    @staticmethod
    def _extract_q(qhat_values: QHatValues) -> EDivisiveChangePoint:
        """
        Given an ordered sequence of Q-Hat values, output the max value and index.

        :param qhat_values: qhat values
        :return: An EDivisiveChangePoint.
        """
        if qhat_values.values.size:
            max_q_index = np.argmax(qhat_values.values)
            max_q = qhat_values.values[max_q_index]
        else:
            max_q = 0
            max_q_index = 0

        return EDivisiveChangePoint(
            index=max_q_index,
            value=max_q,
            value_to_avg=EDivisiveNumpyImplementation._ratio(max_q, qhat_values.average_value),
            value_to_avg_diff=EDivisiveNumpyImplementation._ratio(max_q, qhat_values.average_diff),
            average=qhat_values.average_value,
            average_diff=qhat_values.average_diff,
            window_size=qhat_values.length,
            probability=None,
        )

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        """
        Compute a ratio and handle zero denominator values.

        :param numerator: the ratio numerator.
        :param denominator: the ratio denominator.
        :return: the ratio or NaN if the denominator is equal to zero.
        """
        return numerator / denominator if denominator != 0 else float("nan")

    @staticmethod
    def calculate_q(
        term1: Union[float, np.ndarray],
        term2: Union[float, np.ndarray],
        term3: Union[float, np.ndarray],
        m: int,
        n: int,
    ) -> Union[float, np.ndarray]:
        """
        Calculate the q value from the terms and coefficients.

        :param term1: The current cumulative value for the first
        term in the E-Divisive algorithm. This is the sum of the differences to
        the right of the current location.
        :param term2: The current cumulative value for the second
        term in the E-Divisive algorithm. This is the sum of the differences to
        the at the current location.
        :param term3: The current cumulative value for the third
        term in the E-Divisive algorithm. This is the sum of the differences to
        the left of the current location.
        :param m: The current row location.
        :param n: The current column location.

        :return: The q value generated from the terms.
        """
        term1_reg = term1 * (2.0 / (m * n))
        term2_reg = term2 * (2.0 / (n * (n - 1)))
        term3_reg = term3 * (2.0 / (m * (m - 1)))
        newq = (m * n // (m + n)) * (term1_reg - term2_reg - term3_reg)
        return newq

    @staticmethod
    def calculate_diffs(series: np.ndarray) -> np.ndarray:
        """
        Given an array N calculate an NxN difference matrix.

        :param series: The array to calculate the matrix for.
        :return: The difference matrix.
        """
        row, col = np.meshgrid(series, series)
        diffs = abs(row - col)
        return diffs

    def qhat_values(self, series: np.ndarray) -> np.ndarray:
        """
        Check the input values, calculate the diffs matrix and delegate to calculate_qhat_values.
        Implements change-point detection algorithm from https://arxiv.org/pdf/1306.4933.pdf.

        :param series: the points to process
        :return: The qhat values.
        """

        return self._qhat_values(series).values

    def _qhat_values(self, series: np.ndarray) -> QHatValues:
        """
        Calculate the diffs matrix and delegate to calculate_qhat_values.

        :param series: the points to process
        :return: The qhat values.
        """

        # used as the window size in extract_q
        length = len(series)
        qhat_values = np.zeros(length, dtype=np.float)
        if length < 5:
            # Average value and average diff are used even when there is no data.
            # This avoids an error.
            average_value = 1
            average_diff = 1
            return QHatValues(qhat_values, average_value, average_diff, length)

        return self._calculate_qhat_values(series, None, qhat_values)

    def _calculate_qhat_values(
        self, series: np.ndarray, diffs: np.ndarray, qhat_values: np.ndarray
    ) -> QHatValues:
        """
        Find Q-Hat values for all candidate change points. This provides the current
        'best' python implementation. The intention is to override this for other
        implementations, say a native implementation.

        :param series: The points to process.
        :param diffs: The matrix of diffs.
        :param qhat_values: The array to store the qhat values.
        :return: The qhat values.
        """
        diffs = self.calculate_diffs(series)

        average_value = np.average(series)
        average_diff = np.average(diffs)

        length = len(series)
        n = 2
        m = length - n

        # Each line is preceded by the equivalent list comprehension.

        # term1 = sum(diffs[i][j] for i in range(n) for j in range(n, self.window))
        # See e_divisive.md
        term1 = np.sum(diffs[:n, n:])

        # term2 = sum(diffs[i][k] for i in range(n) for k in range(i + 1, n)) # See e_divisive.md
        term2 = np.sum(np.triu(diffs[:n, :n], 0))

        # term3 = sum(diffs[j][k] for j in range(n, self.window)
        #                         for k in range(j + 1, self.window)) # See e_divisive.md
        term3 = np.sum(np.triu(diffs[n:, n + 1 :], 0))

        qhat_values[n] = self.calculate_q(term1, term2, term3, m, n)

        for n in range(3, (length - 2)):
            m = length - n
            column_delta = np.sum(diffs[n - 1, : n - 1])
            row_delta = np.sum(diffs[n:, n - 1])

            term1 = term1 - column_delta + row_delta
            term2 = term2 + column_delta
            term3 = term3 - row_delta

            qhat_values[n] = self.calculate_q(term1, term2, term3, m, n)

        return QHatValues(qhat_values, average_value, average_diff, length)

    def compute_change_points(
        self, series: List[float], seed: int = 1234
    ) -> List[EDivisiveChangePoint]:
        """
        Compute change points for a series of values.

        :raises: FloatingPointError for numpy errors.
        :see: 'numpy.seterr
        <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.seterr.html>'
        :see: 'numpy.errstate
        <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.errstate.html>'
        """
        if series is None:
            series = []
        if not isinstance(series, np.ndarray) or not isinstance(series.dtype, np.float):
            series = np.array(series, np.float)

        with deterministic_random(seed), np.errstate(all="raise"):
            return self._compute_change_points(series)

    def _compute_change_points(self, series: np.ndarray) -> List[EDivisiveChangePoint]:
        """
        Compute change points over a series of floats.
        :param series: The series.
        :return: The change points.
        """
        LOG.info("compute_change_points")
        pts = len(series)
        qhat_values = self._qhat_values(series)
        LOG.debug("compute_change_points", qs=qhat_values, series=series)
        first_q = self._extract_q(qhat_values)
        max_q = first_q.value
        min_change = max_q
        change_points: List[EDivisiveChangePoint] = []

        # HIERARCHICALLY COMPUTE OTHER CHANGEPOINTS
        terminated = False
        while not terminated:
            candidates = []
            windows = [0] + sorted([c[0] for c in change_points]) + [pts]
            LOG.debug("compute_change_points", windows=windows)
            for i in range(len(windows) - 1):
                window = series[windows[i] : windows[i + 1]]
                win_qs = self._qhat_values(window)
                win_max = self._extract_q(win_qs)
                win_max = win_max._replace(index=win_max.index + windows[i])
                candidates.append(win_max)
                LOG.debug(
                    "compute_change_points candidate", win_qs=win_qs, series=window, win_max=win_max
                )
            candidates.sort(key=lambda tup: tup[1])
            candidate_q = candidates[len(candidates) - 1][1]
            LOG.debug("compute_change_points", candidate_q=candidate_q)

            # RANDOMLY PERMUTE CLUSTERS FOR SIGNIFICANCE TEST

            above = 0.0  # results from permuted test >= candidate_q
            for i in range(self.permutations):
                permute_candidates = []
                for j in range(len(windows) - 1):
                    window = copy.copy(series[windows[j] : windows[j + 1]])
                    # Backwards compatibility with Python 2.7 implementation, see:
                    # https://stackoverflow.com/questions/38943038/difference-between-python-2-and-3-for-shuffle-with-a-given-seed
                    random.shuffle(window, random.random)
                    win_qs = self._qhat_values(window)
                    win_max = self._extract_q(win_qs)
                    win_max_tup = (win_max.index + windows[j], win_max.value)
                    permute_candidates.append(win_max_tup)
                    LOG.debug(
                        "compute_change_points", candidate_q=candidate_q, candidates=candidates
                    )
                permute_candidates.sort(key=lambda tup: tup[1])
                permute_q = permute_candidates[len(permute_candidates) - 1][1]
                LOG.debug("compute_change_points", permute_q=permute_q)
                if permute_q >= candidate_q:
                    above += 1

            # for coloring the lines, we will use the first INSIGNIFICANT point
            # as our baseline for transparency
            if candidate_q < min_change:
                min_change = candidate_q

            probability = above / (self.permutations + 1)
            if probability > self.pvalue:
                terminated = True
            else:
                change_points.append(candidates[-1]._replace(probability=probability))

        LOG.debug("compute_change_points", change_points=change_points)
        return change_points
