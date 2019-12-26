"""
Change points detection related tests.
"""
import unittest
import numpy as np
from signal_processing_algorithms.e_divisive import EDivisive
from signal_processing_algorithms.e_divisive_numpy import EDivisiveChangePoint


class TestPostRunCheck(unittest.TestCase):
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
        pvalue = 0.01
        permutations = 100
        algo = EDivisive(pvalue, permutations)
        points = algo.compute_change_points(series)

        assert 3 == len(points)

        expected = EDivisiveChangePoint(
            index=40,
            window_size=60,
            value_to_avg=26.864695599042758,
            average_diff=88.77666666666667,
            average=103.36666666666666,
            value=2776.9140350877196,
            value_to_avg_diff=31.279773608918102,
            probability=0.0,
        )

        assert expected == points[0]

        expected = EDivisiveChangePoint(
            index=20,
            window_size=40,
            value_to_avg=16.517170931024417,
            average_diff=42.9325,
            average=54.1,
            value=893.578947368421,
            value_to_avg_diff=20.813578230208375,
            probability=0.0,
        )

        assert expected == points[1]

        expected = EDivisiveChangePoint(
            index=50,
            window_size=20,
            value_to_avg=0.8304441142479775,
            average_diff=36.06,
            average=201.9,
            value=167.66666666666666,
            value_to_avg_diff=4.649657977444998,
            probability=0.009900990099009901,
        )

        assert expected == points[2]

    def _test_helper(self, series=None):
        """
        Helper for simple regression test.
        """
        if series is None:
            series = np.full(30, 50, dtype=np.int)
            series[15:30] = 100
        pvalue = 0.01
        permutations = 100

        algo = EDivisive(pvalue, permutations)
        points = algo.compute_change_points(series)
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
        kwargs = {
            "index": 15,
            "window_size": 30,
            "value_to_avg": 7.913043478260868,
            "average_diff": 24.88888888888889,
            "average": 76.66666666666667,
            "value": 606.6666666666666,
            "value_to_avg_diff": 24.374999999999996,
            "probability": 0.0,
        }
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
        kwargs = {
            "index": 15,
            "window_size": 33,
            "value_to_avg": 7.030797385620915,
            "average_diff": 24.977043158861342,
            "average": 75.75757575757575,
            "value": 532.636165577342,
            "value_to_avg_diff": 21.325028835063435,
            "probability": 0.0,
        }
        expected = EDivisiveChangePoint(**kwargs)
        assert points[0] == expected

        kwargs = {
            "index": 33,
            "window_size": 45,
            "value_to_avg": 2.9912023460410566,
            "average_diff": 23.506172839506174,
            "average": 68.88888888888889,
            "value": 206.06060606060612,
            "value_to_avg_diff": 8.766233766233768,
            "probability": 0.0,
        }
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

        kwargs = {
            "index": 15,
            "window_size": 30,
            "value_to_avg": 7.913043478260868,
            "average_diff": 24.88888888888889,
            "average": 76.66666666666667,
            "value": 606.6666666666666,
            "value_to_avg_diff": 24.374999999999996,
            "probability": 0.0,
        }
        expected = EDivisiveChangePoint(**kwargs)
        assert expected == points[0]

        kwargs = {
            "index": 30,
            "window_size": 45,
            "value_to_avg": 11.959075407351268,
            "average_diff": 43.65432098765432,
            "average": 101.11111111111111,
            "value": 1209.1954022988505,
            "value_to_avg_diff": 27.699329068497423,
            "probability": 0.0,
        }
        expected = EDivisiveChangePoint(**kwargs)
        assert expected == points[1]

    def test_no_regressions(self):
        """
        Test no regression.
        """
        series = np.full(30, 50, dtype=np.int)
        points, _ = self._test_helper(series=series)
        assert 0 == len(points)
