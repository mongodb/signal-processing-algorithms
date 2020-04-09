import numpy as np
from signal_processing_algorithms.e_divisive.base import EDivisiveCalculator
from signal_processing_algorithms.e_divisive.calculators import numpy_calculator as EDivisive


class NumpyWindowedCalculator(EDivisiveCalculator):

    @staticmethod
    def calculate_diffs(series: np.ndarray) -> np.ndarray:
        return EDivisive.calculate_diffs(series)

    @staticmethod
    def calculate_qhat_values(diffs: np.ndarray) -> np.ndarray:
        length = len(diffs)
        qhat_values = np.zeros(len(diffs), dtype=np.float)
        if length < 5:
            return qhat_values
        window = int(round(length / 2))

        for n in range(2, length - 2):
            m = length - n

            term1 = np.sum(diffs[max(0, n - window + 1): n, n: min(length, n + window)])

            row = max(n - window + 1, 0)
            column = row + min(window - 2 + 1, n)
            term2 = np.sum(np.triu(diffs[row:column, row:column], 1))

            term3 = np.sum(np.triu(diffs[n: window + n + 1, n: window + n + 1], 1))

            qhat_values[n] = EDivisive._calculate_q(term1, term2, term3, m, n)
        return qhat_values
