import numpy as np
import pytest

from signal_processing_algorithms.distance import get_distance_matrix


class TestDifference(object):
    def test_get_distance_matrix_invalid_input_type(self):
        """Test invalid input type - List."""
        x = [i for i in range(10)]
        with pytest.raises(TypeError):
            get_distance_matrix(x, x)

    def test_get_distance_matrix_invalid_input_array(self):
        """Test invalid input array."""
        x = np.asarray([i for i in range(10)])
        y = np.asarray([i for i in range(10, 14)])
        with pytest.raises(np.AxisError):
            get_distance_matrix(x, y)

    def test_get_distance_matrix_same_distribution(self):
        """Test distance matrix for same distribution."""
        x = np.asarray([[0], [1], [2], [3], [4]])
        expected = np.asarray(
            [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]
        )
        np.testing.assert_equal(get_distance_matrix(x, x), expected)

    def test_get_distance_matrix_two_distributions(self):
        """Test distance matrix for different distributions."""
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        y = np.asarray([[10, 20], [15, 17], [12, 14], [9, 10]])
        expected = np.asarray(
            [
                [20.1246118, 20.51828453, 16.2788206, 11.3137085],
                [17.4642492, 17.69180601, 13.45362405, 8.48528137],
                [14.86606875, 14.86606875, 10.63014581, 5.65685425],
            ]
        )
        np.testing.assert_almost_equal(get_distance_matrix(x, y), expected)
