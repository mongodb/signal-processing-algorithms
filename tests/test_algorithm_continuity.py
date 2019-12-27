"""
E-Divisive related tests.
"""


import numpy as np
from signal_processing_algorithms.e_divisive import EDivisive
from signal_processing_algorithms.e_divisive import __name__ as under_test_module_name
from miscutils.testing import relative_patch_maker

patch = relative_patch_maker(under_test_module_name)


class OldEDivisive(object):
    """
    This is the original O(n^2) E-Divisive implementation as described in the whitepaper.
    It is here for comparison purposes only and to allow the q values to
    be generated if further tests are added.

    NOTE: This is why I have disabled some pylint checks.
    NOTE: This implementation is purely to provide a 'canonical' implementation for
    test purposes. It is not efficient and will not be optimized.
    """

    # Implementing change-point detection algorithm from https://arxiv.org/pdf/1306.4933.pdf
    def qs(self, series: np.ndarray):
        """
        Find Q-Hat values for all candidate change points

        :param series: the points to process
        :return:
        """
        length = len(series)
        qs = np.zeros(length, dtype=np.float)
        if length < 5:
            return qs

        diffs = [[abs(series[i] - series[j]) for i in range(length)] for j in range(length)]

        for n in range(2, length - 2):
            m = length - n

            term1 = sum(diffs[i][j] for i in range(n) for j in range(n, length))
            term2 = sum(diffs[i][k] for i in range(n) for k in range(i + 1, n))
            term3 = sum(diffs[j][k] for j in range(n, length) for k in range(j + 1, length))

            term1_reg = term1 * (2.0 / (m * n))
            term2_reg = term2 * (2.0 / (n * (n - 1)))
            term3_reg = term3 * (2.0 / (m * (m - 1)))
            newq = (m * n // (m + n)) * (term1_reg - term2_reg - term3_reg)
            qs[n] = newq

        return qs


class TestAlgorithmContinuity(object):
    """
    Test PERF-1635 is fixed correctly.
    """

    series = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3], dtype=np.float)
    expected = np.array(
        [
            0,
            0,
            1.3777777777777778,
            3.4444444444444438,
            4.428571428571429,
            2.971428571428571,
            3.599999999999999,
            2.342857142857143,
            2.857142857142857,
            4.666666666666666,
            0,
            0,
        ],
        dtype=np.float,
    )

    def test_old_algorithm(self):
        """
        Test to double check slow O(n^2) algorithm. Small data set so this is ok.
        """
        algorithm = OldEDivisive()
        q_values = algorithm.qs(self.series)
        assert all(np.isclose(self.expected, q_values))

    def test_fixed(self):
        """
        Test that the current algorithm generates the same q values as the original.
        """
        algorithm = EDivisive()
        q_values = algorithm.qhat_values(self.series)
        assert all(np.isclose(self.expected, q_values))

    @patch("e_divisive_native_wrapper")
    def test_fallback(self, mock_native_wrapper):
        """
        Test that the fallback algorithm generates the same q values as the original.
        """
        mock_native_wrapper.LOADED = False
        algorithm = EDivisive()
        q_values = algorithm.qhat_values(self.series)
        assert all(np.isclose(self.expected, q_values))


class TestRobustContinuity:
    """
    Robust test for PERF-1635.
    """

    def test_old_algorithm(self, robust_series, expected_result_robust_series):
        """
        Test to double check slow O(n^2) algorithm. Small data set so this is ok.
        """
        algorithm = OldEDivisive()
        q_values = algorithm.qs(robust_series)
        assert all(np.isclose(expected_result_robust_series, q_values))

    def test_q_values(self, robust_series, expected_result_robust_series):
        """
        Test that the current algorithm generates the same q values as the original.
        """
        algorithm = EDivisive()
        q_values = algorithm.qhat_values(robust_series)
        assert all(np.isclose(expected_result_robust_series, q_values))

    @patch("e_divisive_native_wrapper")
    def test_fallback(self, mock_native_wrapper, robust_series, expected_result_robust_series):
        """
        Test that the fallback algorithm generates the same q values as the original.
        """
        mock_native_wrapper.LOADED = False
        algorithm = EDivisive()
        q_values = algorithm.qhat_values(robust_series)
        assert all(np.isclose(expected_result_robust_series, q_values))
