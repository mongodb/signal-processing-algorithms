"""E-Divisive Numpy Calculator."""
import numpy as np


def _calculate_q(
    term1: np.ndarray, term2: np.ndarray, term3: np.ndarray, m: int, n: int
) -> np.ndarray:
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


def calculate_diffs(series: np.ndarray) -> np.ndarray:
    """
    Given an array N calculate an NxN difference matrix.

    :param series: The array to calculate the matrix for.
    :return: The difference matrix.
    """
    row, col = np.meshgrid(series, series)
    diffs = abs(row - col)
    return diffs


def calculate_qhat_values(diffs: np.ndarray) -> np.ndarray:
    """
    Find Q-Hat values given a difference matrix.

    :param diffs: The difference matrix.
    :return: The qhat values.
    """
    length = len(diffs)
    qhat_values = np.zeros(length, dtype=np.float)

    if length < 5:
        return qhat_values

    n = 2
    m = length - n

    # Each line is preceded by the equivalent list comprehension.

    # term1 = sum(diffs[i][j] for i in range(n) for j in range(n, self.window))
    term1 = np.sum(diffs[:n, n:])

    # term2 = sum(diffs[i][k] for i in range(n) for k in range(i + 1, n))
    term2 = np.sum(np.triu(diffs[:n, :n], 0))

    # term3 = sum(diffs[j][k] for j in range(n, self.window)
    #                         for k in range(j + 1, self.window))
    term3 = np.sum(np.triu(diffs[n:, n + 1 :], 0))

    qhat_values[n] = _calculate_q(term1, term2, term3, m, n)

    for n in range(3, (length - 2)):
        m = length - n
        column_delta = np.sum(diffs[n - 1, : n - 1])
        row_delta = np.sum(diffs[n:, n - 1])

        term1 = term1 - column_delta + row_delta
        term2 = term2 + column_delta
        term3 = term3 - row_delta

        qhat_values[n] = _calculate_q(term1, term2, term3, m, n)

    return qhat_values
