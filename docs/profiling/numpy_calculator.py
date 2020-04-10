import numpy as np
from signal_processing_algorithms.e_divisive.base import EDivisiveCalculator
from signal_processing_algorithms.e_divisive.calculators import numpy_calculator as EDivisive


class NumpyCalculator(EDivisiveCalculator):
    """
    Numpy O(n^2) implementation. Only used for profiling, don't use this in production!
    """

    @staticmethod
    def calculate_diffs(series: np.ndarray) -> np.ndarray:
        return EDivisive.calculate_diffs(series)

    @staticmethod
    def calculate_qhat_values(diffs: np.ndarray) -> np.ndarray:
        length = len(diffs)
        qhat_values = np.zeros(len(diffs), dtype=np.float)
        if length < 5:
            return qhat_values

        for n in range(2, length - 2):
            m = length - n

            term1 = np.sum(diffs[:n, n:])
            term2 = np.sum(np.triu(diffs[:n, :n], 0))
            term3 = np.sum(np.triu(diffs[n:, n + 1:], 0))

            qhat_values[n] = EDivisive._calculate_q(term1, term2, term3, m, n)
        return qhat_values
