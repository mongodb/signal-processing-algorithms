import numpy as np
from signal_processing_algorithms.e_divisive.base import EDivisiveCalculator
from signal_processing_algorithms.e_divisive import numpy_calculator as EDivisive


class OriginalCalculator(EDivisiveCalculator):
    """Only used for profiling, don't use this in production!"""

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
            term1 = sum(diffs[i][j] for i in range(n) for j in range(n, length))
            term2 = sum(diffs[i][k] for i in range(n) for k in range(i + 1, n))
            term3 = sum(diffs[j][k] for j in range(n, length) for k in range(j + 1, length))

            qhat_values[n] = EDivisive._calculate_q(term1, term2, term3, m, n)

        return qhat_values
