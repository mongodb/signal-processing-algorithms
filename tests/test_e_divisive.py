"""
E-Divisive related tests.
"""

from signal_processing_algorithms.energy_statistics.energy_statistics import e_divisive


class TestEDivisive(object):
    """
    Test for EDivisive methods.
    """

    def test_absolute_certainty_univariate(self):
        series = [0] * 1000 + [1] * 1000
        res = e_divisive(series)
        assert res == [1000]

    def test_absolute_certainty_multivariate(self):
        series = [[0, 0]] * 100 + [[1, 1]] * 100 + [[20, 20]] * 100
        res = e_divisive(series)
        assert res == [200, 100]
