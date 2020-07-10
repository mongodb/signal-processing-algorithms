"""E-Divisive Numpy Calculator."""
import numpy as np

from scipy.special import comb


def _calculate_q(cross_term: float, x_term: float, y_term: float, x_len: int, y_len: int) -> float:
    """
    Calculate the q value from the terms and coefficients.

    :param cross_term: The sum of differences between sample distributions X, Y.
    :param x_term: The sum of differences within sample distribution X.
    :param y_term: The sum of differences within sample distribution Y.
    :param x_len: The length of sample distribution X.
    :param y_len: The length of sample distribution Y.

    :return: The q value generated from the terms.
    """
    cross_term_reg = 0 if x_len < 1 or y_len < 1 else cross_term * (2.0 / (x_len * y_len))
    x_term_reg = 0 if x_len < 2 else x_term * (comb(x_len, 2) ** -1)
    y_term_reg = 0 if y_len < 2 else y_term * (comb(y_len, 2) ** -1)
    new_q = (x_len * y_len / (x_len + y_len)) * (cross_term_reg - x_term_reg - y_term_reg)
    return new_q


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
    qhat_values = np.zeros(len(diffs), dtype=np.float)

    # We will partition our signal into:
    # X = {Xi; 0 <= i < tau}
    # Y = {Yj; tau <= j < len(signal) }
    # and look for argmax(tau)Q(tau)

    # sum |Xi - Yj| for i < tau <= j
    cross_term = 0
    # sum |Xi - Xj| for i < j < tau
    x_term = 0
    # sum |Yi - Yj| for tau <= i < j
    y_term = 0

    for row in range(0, len(diffs)):
        y_term += np.sum(diffs[row, row:])

    for tau in range(0, len(diffs)):
        qhat_values[tau] = _calculate_q(cross_term, x_term, y_term, tau, len(diffs) - tau)

        column_delta = np.sum(diffs[:tau, tau])
        row_delta = np.sum(diffs[tau, tau:])

        cross_term = cross_term - column_delta + row_delta
        x_term = x_term + column_delta
        y_term = y_term - row_delta

    return qhat_values
