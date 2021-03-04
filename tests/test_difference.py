import numpy as np
import pytest

from signal_processing_algorithms.difference import pairwise_difference


class TestDifference(object):
    def test_pairwise_distance_invalid_input_type(self):
        x = [i for i in range(10)]
        with pytest.raises(TypeError):
            pairwise_difference(x)

    def test_pairwise_distance_one_distribution(self):
        x = np.arange(0, 5, 1)
        expected = np.asarray(
            [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]
        )
        np.testing.assert_equal(pairwise_difference(x), expected)

    def test_pairwise_distance_two_distributions(self):
        x = np.arange(10, 20, 2)
        y = np.arange(1, 7, 1)
        expected = [
            [9, 8, 7, 6, 5, 4],
            [11, 10, 9, 8, 7, 6],
            [13, 12, 11, 10, 9, 8],
            [15, 14, 13, 12, 11, 10],
            [17, 16, 15, 14, 13, 12],
        ]
        np.testing.assert_equal(pairwise_difference(x, y), expected)
