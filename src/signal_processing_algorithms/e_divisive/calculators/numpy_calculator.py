"""E-Divisive Numpy Calculator."""
import numpy as np


def _calculate_q(
        cross_term: np.ndarray, x_term: np.ndarray, y_term: np.ndarray, x_len: int, y_len: int
) -> np.ndarray:
    """
    Calculate the q value from the terms and coefficients.

    :param cross_term: The sum of differences between sample distributions X, Y.
    :param x_term: The sum of differences within sample distribution X.
    :param y_term: The sum of differences within sample distribution Y.
    :param x_len: The length of sample distribution X.
    :param y_len: The length of sample distribution Y.

    :return: The q value generated from the terms.
    """
    term1_reg = cross_term * (2.0 / (x_len * y_len))
    term2_reg = x_term * (2.0 / (y_len * (y_len - 1)))
    term3_reg = y_term * (2.0 / (x_len * (x_len - 1)))
    newq = (x_len * y_len // (x_len + y_len)) * (term1_reg - term2_reg - term3_reg)
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
    qhat_values = np.zeros(len(diffs), dtype=np.float)

    # We will partition our signal into:
    # X = {Xi; 0 <= i < tau}
    # Y = {Yj; tau <= j < len(signal) }
    # and look for argmax(tau)Q(tau)
    tau = 0

    # sum |Xi - Yj| for i < tau <= j
    cross_term = np.sum(diffs[:tau, tau:])
    # sum |Xi - Xj| for i < j < tau
    x_term = np.sum(np.triu(diffs[:tau, :tau]))
    # sum |Yi - Yj| for tau <= i < j
    y_term = np.sum(np.triu(diffs[tau:, tau:]))

    qhat_values[tau] = _calculate_q(cross_term, x_term, y_term, tau, len(diffs) - tau)

    for tau in range(1, len(diffs)):
        column_delta = np.sum(diffs[tau, :tau])
        row_delta = np.sum(diffs[tau:, tau])

        cross_term = cross_term - column_delta + row_delta
        x_term = x_term + column_delta
        y_term = y_term - row_delta

        qhat_values[tau] = _calculate_q(cross_term, x_term, y_term, tau, len(diffs) - tau)

    return qhat_values
