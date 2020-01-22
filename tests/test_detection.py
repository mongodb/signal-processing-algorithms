"""
Change points detection related tests.
"""
import numpy as np
import pytest

from signal_processing_algorithms.e_divisive import EDivisive
from signal_processing_algorithms.e_divisive.calculators import cext_calculator
from signal_processing_algorithms.e_divisive.change_points import EDivisiveChangePoint
from signal_processing_algorithms.e_divisive.significance_test import (
    QHatPermutationsSignificanceTester,
)


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
        calculator = cext_calculator
        tester = QHatPermutationsSignificanceTester(
            calculator=calculator, pvalue=0.01, permutations=100
        )
        algo = EDivisive(seed=1234, calculator=calculator, significance_tester=tester)
        points = algo.get_change_points(series)

        assert 3 == len(points)

        expected = EDivisiveChangePoint(index=40, qhat=2776.9140350877196, probability=0.0)

        assert expected == points[0]

        expected = EDivisiveChangePoint(index=20, qhat=893.578947368421, probability=0.0)

        assert expected == points[1]

        expected = EDivisiveChangePoint(
            index=50, qhat=167.66666666666666, probability=0.009900990099009901
        )

        assert expected == points[2]

    def _test_helper(self, series=None):
        """
        Helper for simple regression test.
        """
        if series is None:
            series = np.full(30, 50, dtype=np.int)
            series[15:30] = 100
        calculator = cext_calculator
        tester = QHatPermutationsSignificanceTester(
            pvalue=0.01, permutations=100, calculator=calculator
        )
        algo = EDivisive(seed=1234, calculator=calculator, significance_tester=tester)
        points = algo.get_change_points(series)
        points = sorted(points, key=lambda point: point.index)
        return points, series

    def test_finds_simple_regression(self):
        """
        Test finding a simple regression.
        """
        points, state = self._test_helper()

        assert 1 == len(points)
        assert 15 == points[0].index

    def test_finds_ahead(self):
        """
        Test ahead.
        """
        series = [50] * 14 + [74] + [100] * 15
        points, state = self._test_helper(series=series)

        assert 1 == len(points)
        assert 15 == points[0].index

    def test_finds_behind(self):
        """
        Test finding behind.
        """
        series = np.full(30, 50, dtype=np.int)
        series[14] = 76
        series[15:30] = 100

        points, state = self._test_helper(series=series)

        assert 1 == len(points)
        assert 14 == points[0].index

    def test_finds_simple_regression2(self):
        """
        Test another simple regression.
        """
        series = np.full(30, 50, dtype=np.int)
        series[2] = 100
        series[15:30] = 100

        points, state = self._test_helper(series=series)

        assert 1 == len(points)
        kwargs = {"index": 15, "qhat": 606.6666666666666, "probability": 0.0}
        expected = EDivisiveChangePoint(**kwargs)
        assert expected == points[0]

    def test_regression_and_recovery(self):
        """
        Test regression and recovery.
        """
        # create an array filled with 50s' then set some ranges to 100
        series = np.full(45, 50, dtype=np.int)
        series[2] = 100
        series[15:30] = 100
        series[32] = 100

        points, state = self._test_helper(series=series)
        assert 2 == len(points)
        kwargs = {"index": 15, "qhat": 532.636165577342, "probability": 0.0}
        expected = EDivisiveChangePoint(**kwargs)
        assert points[0] == expected

        kwargs = {"index": 33, "qhat": 206.06060606060612, "probability": 0.0}
        expected = EDivisiveChangePoint(**kwargs)
        assert points[1] == expected

    def test_two_regressions(self):
        """
        Test 2 regressions.
        """
        length = 45

        series = np.full(length, 50, dtype=np.int)
        series[2] = 100
        series[15:30] = 100
        series[30:] = 150

        points, state = self._test_helper(series=series)

        assert 2 == len(points)

        kwargs = {"index": 15, "qhat": 606.6666666666666, "probability": 0.0}
        expected = EDivisiveChangePoint(**kwargs)
        assert expected == points[0]

        kwargs = {"index": 30, "qhat": 1209.1954022988505, "probability": 0.0}
        expected = EDivisiveChangePoint(**kwargs)
        assert expected == points[1]

    def test_no_regressions(self):
        """
        Test no regression.
        """
        series = np.full(30, 50, dtype=np.int)
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

    def _test_helper(self, expected, series):
        """
        Helper for simple regression test.
        """
        calculator = cext_calculator
        tester = QHatPermutationsSignificanceTester(
            pvalue=0.01, permutations=100, calculator=calculator
        )
        algo = EDivisive(seed=1234, calculator=calculator, significance_tester=tester)
        points = algo.get_change_points(series)
        points = sorted(points, key=lambda point: point.index)
        assert expected == len(points)

    def test_short(self, short_profile):
        series = short_profile['series']
        expected = short_profile['expected']
        self._test_helper(expected, series)

    def test_small(self, small_profile):
        series = small_profile['series']
        expected = small_profile['expected']
        self._test_helper(expected, series)

    def test_medium(self, medium_profile):
        series = medium_profile['series']
        expected = medium_profile['expected']
        self._test_helper(expected, series)

    # takes ~9 seconds on laptop
    @pytest.mark.slow
    def test_large(self, large_profile):
        series = large_profile['series']
        expected = large_profile['expected']
        self._test_helper(expected, series)

    # takes around 90 seconds
    @pytest.mark.slow
    def test_very_large(self, very_large_profile):
        series = very_large_profile['series']
        expected = very_large_profile['expected']
        self._test_helper(expected, series)

    # takes around 15 minutes
    @pytest.mark.slow
    def test_huge(self, huge_profile):
        series = huge_profile['series']
        expected = huge_profile['expected']
        self._test_helper(expected, series)

    # takes around 2h 15m
    @pytest.mark.slow
    def test_humungous(self, humongous_profile):
        series = humongous_profile['series']
        expected = humongous_profile['expected']
        self._test_helper(expected, series)
