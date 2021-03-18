"""
Change points detection related tests.
"""
import numpy as np
import pytest

from signal_processing_algorithms.determinism import deterministic_numpy_random
from signal_processing_algorithms.energy_statistics.energy_statistics import e_divisive


class TestPostRunCheck:
    """
    Test post run check.
    """

    NUMBER_TYPES = (int, int, float, complex)

    def test_random_ish_data(self):
        """
        A cheeky test that noisy-looking data (random numbers generated from disjoint
        intervals) finds regressions. More of a regression-check than anything.
        """
        # from random import *
        # series = [ randint(150, 250) for _ in range(0,20) ]
        # print(series)
        # randint(1,50)
        series = [
            41,
            18,
            23,
            3,
            32,
            11,
            40,
            13,
            29,
            48,
            47,
            35,
            18,
            21,
            6,
            2,
            23,
            3,
            4,
            7,
            # randint(60,120)
            120,
            103,
            102,
            81,
            71,
            62,
            115,
            61,
            108,
            63,
            70,
            98,
            65,
            96,
            64,
            74,
            70,
            113,
            90,
            114,
            # randint(150,250)
            208,
            196,
            153,
            150,
            225,
            179,
            206,
            165,
            177,
            151,
            218,
            217,
            244,
            245,
            229,
            195,
            225,
            229,
            176,
            250,
        ]

        with deterministic_numpy_random(1234):
            points = e_divisive(series, pvalue=0.01)
            assert points == [40, 20]

    def _test_helper(self, series=None):
        """
        Helper for simple regression test.
        """
        if series is None:
            series = np.full(30, 50, dtype=int)
            series[15:30] = 100
        with deterministic_numpy_random(seed=1234):
            points = e_divisive(series, pvalue=0.01)
        points = sorted(points)
        return points, series

    def test_finds_simple_regression(self):
        """
        Test finding a simple regression.
        """
        points, state = self._test_helper()

        assert points == [15]

    def test_finds_simple_regression_multivariate(self):
        """
        Test finding a simple regression for multivariate series.
        """
        series = np.full(30, 50, dtype=int)
        series[15:30] = 100
        series = series.reshape(10, 3)
        points, state = self._test_helper(series=series)

        assert points == [5]

    def test_finds_ahead(self):
        """
        Test ahead.
        """
        series = [50] * 14 + [74] + [100] * 15
        points, state = self._test_helper(series=series)

        assert points == [15]

    def test_finds_ahead_multivariate(self):
        """
        Test ahead.
        """
        series = [[50, 50]] * 14 + [[74, 74]] + [[100, 100]] * 15
        points, state = self._test_helper(series=series)

        assert points == [15]

    def test_finds_behind(self):
        """
        Test finding behind.
        """
        series = np.full(30, 50, dtype=int)
        series[14] = 76
        series[15:30] = 100

        points, state = self._test_helper(series=series)

        assert points == [14]

    def test_finds_behind_multivariate(self):
        """
        Test finding behind.
        """
        series = [[50, 50]] * 14 + [[76, 76]] + [[100, 100]] * 15

        points, state = self._test_helper(series=series)

        assert points == [14]

    def test_finds_simple_regression2(self):
        """
        Test another simple regression.
        """
        series = np.full(30, 50, dtype=int)
        series[2] = 100
        series[15:30] = 100

        points, state = self._test_helper(series=series)

        assert points == [15]

    def test_finds_simple_regression2_multivariate(self):
        """
        Test another simple regression.
        """
        series = [[50, 50]] * 2 + [[100, 100]] + [[50, 50]] * 12 + [[100, 100]] * 15

        points, state = self._test_helper(series=series)

        assert points == [15]

    def test_regression_and_recovery(self):
        """
        Test regression and recovery.
        """
        # create an array filled with 50s' then set some ranges to 100
        series = np.full(45, 50, dtype=int)
        series[2] = 100
        series[15:30] = 100
        series[32] = 100

        with deterministic_numpy_random(1000):
            points = e_divisive(series, pvalue=0.01)
            assert points == [33, 15]

    def test_two_regressions(self):
        """
        Test 2 regressions.
        """
        length = 45

        series = np.full(length, 50, dtype=int)
        series[2] = 100
        series[15:30] = 100
        series[30:] = 150

        points, state = self._test_helper(series=series)

        assert points == [15, 30]

    def test_two_regressions_multivariate(self):
        """
        Test 2 regressions.
        """

        series = (
            [[50, 50]] * 2 + [[100, 100]] + [[50, 50]] * 12 + [[100, 100]] * 15 + [[150, 150]] * 15
        )

        points, state = self._test_helper(series=series)

        assert points == [15, 30]

    def test_no_regressions(self):
        """
        Test no regression.
        """
        series = np.full(30, 50, dtype=int)
        points, _ = self._test_helper(series=series)
        assert 0 == len(points)

    def test_no_regressions_multivariate(self):
        """
        Test no regression.
        """
        series = np.full(30, 50, dtype=int)
        series.reshape(10, 3)
        points, _ = self._test_helper(series=series)
        assert 0 == len(points)

    def test_long_series(self, long_series):
        """
        Test no regression.
        """
        points, _ = self._test_helper(series=long_series)
        assert 4 == len(points)


class TestComputeChangePoints:
    """
    Test suite for the PointsModel.compute_change_points class.
    """

    def _test_helper(self, expected, series, seed=1234):
        """
        Helper for simple regression test.
        """
        with deterministic_numpy_random(seed):
            points = e_divisive(series, permutations=100, pvalue=0.01)
        points = sorted(points)
        assert expected == len(points)

    def test_short(self, short_profile):
        series = short_profile["series"]
        expected = short_profile["expected"]
        self._test_helper(expected, series)

    def test_small(self, small_profile):
        series = small_profile["series"]
        expected = small_profile["expected"]
        self._test_helper(expected, series)

    def test_medium(self, medium_profile):
        series = medium_profile["series"]
        expected = medium_profile["expected"]
        self._test_helper(expected, series)

    # takes ~9 seconds on laptop
    @pytest.mark.slow
    def test_large(self, large_profile):
        series = large_profile["series"]
        expected = large_profile["expected"]
        self._test_helper(expected, series, seed=100)

    # takes around 90 seconds
    @pytest.mark.slow
    def test_very_large(self, very_large_profile):
        series = very_large_profile["series"]
        expected = very_large_profile["expected"]
        self._test_helper(expected, series)

    # takes around 15 minutes
    @pytest.mark.slow
    def test_huge(self, huge_profile):
        series = huge_profile["series"]
        expected = huge_profile["expected"]
        self._test_helper(expected, series)

    # takes around 2h 15m
    @pytest.mark.slow
    def test_humungous(self, humongous_profile):
        series = humongous_profile["series"]
        expected = humongous_profile["expected"]
        self._test_helper(expected, series)
