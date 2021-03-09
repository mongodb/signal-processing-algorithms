"""
Unit tests for signal_processing/energy_statistics.py.
"""

import numpy as np
import pytest

from signal_processing_algorithms.energy_statistics import EnergyStatistics, get_valid_input


class TestEnergyStatistics(object):
    """Test for Energy Statistics. """

    def test_no_data(self):
        """Test no data."""
        with pytest.raises(ValueError, match=r"Distribution cannot be empty."):
            EnergyStatistics(None, None)
        with pytest.raises(ValueError, match=r"Distribution cannot be empty."):
            EnergyStatistics([], [])

    def test_invalid_number_of_variables(self):
        """Test invalid number of variables for different observations in the two distributions."""
        with pytest.raises(
            ValueError,
            match=r"Number of variables per observation must be the same for both distributions!",
        ):
            EnergyStatistics([1, 2, 3], [[1, 2], [3, 4]])

        with pytest.raises(
            ValueError,
            match=r"Number of variables per observation must be the same for both distributions!",
        ):
            EnergyStatistics([[1], [2], [3]], [[1, 2]])

    def test_energy_statistics_same_distribution(self):
        """Test energy statistics for same distribution."""
        x = np.arange(10, 70)
        energy_statistics = EnergyStatistics(x, x)
        assert energy_statistics.e == 0
        assert energy_statistics.t == 0
        assert energy_statistics.h == 0

    def test_energy_statistics_similar_distributions(self):
        """Test energy statistics for similar distributions."""
        x = np.arange(1, 100, 1)
        y = np.arange(1, 105, 1)
        energy_statistics = EnergyStatistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 0.160, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 8.136, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.002, decimal=3)

    def test_energy_statistics_different_distributions(self):
        """Test energy statistics for different distributions."""
        x = np.arange(1, 100, 1)
        y = np.arange(10000, 13000, 14)
        energy_statistics = EnergyStatistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 21859.691, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 1481794.709, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.954, decimal=3)

    def test_energy_statistics_univariate_ndarray(self):
        """Test energy statistics for uni-variate distributions."""
        x = np.arange(1, 300)
        y = np.arange(1000, 5000, 10)
        energy_statistics = EnergyStatistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 4257.009, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 728381.015, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.748, decimal=3)

    def test_energy_statistics_univariate_list(self):
        """Test energy statistics for uni-variate distributions."""
        x = [i for i in range(300, 500, 1)]
        y = [i for i in range(400, 900, 11)]
        energy_statistics = EnergyStatistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 268.352, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 10035.943, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.532, decimal=3)

    def test_energy_statistics_multivariate_ndarray(self):
        """Test energy statistics for multivariate distributions."""
        x = [[i, i + 7, i + 4] for i in range(100, 200)]
        y = [[i, i + 2, i + 3] for i in range(100, 400, 12)]
        x = np.asarray(x)
        y = np.asarray(y)
        energy_statistics = EnergyStatistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 137.708, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 2754.172, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.373, decimal=3)

    def test_energy_statistics_multivariate_list(self):
        """Test energy statistics for multivariate distributions."""
        x = [[i, i + 1] for i in range(0, 100)]
        y = [[i, i + 10] for i in range(0, 1000)]
        energy_statistics = EnergyStatistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 775.467, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 70497.082, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.599, decimal=3)

    def test_different_distribution_samples(self):
        """Test to compare energy statistics for samples from same distribution to samples
        from different distributions for sufficiently large samples."""
        num_samples = 1000
        random_state = np.random.RandomState(0)

        expectation = 30
        x = random_state.poisson(expectation, size=num_samples)

        mean, standard_deviation = 1, 0.1
        y = random_state.normal(mean, standard_deviation, size=num_samples)
        z = random_state.normal(mean, standard_deviation, size=num_samples)

        diff_es = EnergyStatistics(x, y)
        similar_es = EnergyStatistics(y, z)

        assert diff_es.e > similar_es.e
        assert diff_es.t > similar_es.t
        assert diff_es.h > similar_es.h

    def test_get_valid_input(self):
        """Test for valid input transformation."""
        x = [i for i in range(10)]
        actual = get_valid_input(x)
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.float64
        assert actual.shape == (10, 1)

    def test_get_statistics_and_probabilities_same_distribution_samples_univariate(self):
        """Test sufficiently large number samples drawn from the same distribution have almost zero H."""
        mean, standard_deviation = 1, 0.1
        num_samples = 1000

        random_state = np.random.RandomState(0)
        x = random_state.normal(mean, standard_deviation, size=num_samples)
        y = random_state.normal(mean, standard_deviation, size=num_samples)

        es = EnergyStatistics(x, y)
        es_test_result = es.get_statistics_and_probabilities(num_tests=100)

        np.testing.assert_almost_equal(es_test_result.e, 0, decimal=3)
        np.testing.assert_almost_equal(es_test_result.e_p, 0, decimal=0)
        np.testing.assert_almost_equal(es_test_result.t, 0.125, decimal=3)
        np.testing.assert_almost_equal(es_test_result.t_p, 0, decimal=0)
        np.testing.assert_almost_equal(es_test_result.h, 0.001, decimal=3)
        np.testing.assert_almost_equal(es_test_result.h_p, 0, decimal=0)

    def test_get_statistics_and_probabilities_same_distribution_samples_multivariate(self):
        """Test sufficiently large number samples drawn from the same distribution have low statistics values."""
        mean = np.zeros(3)
        mean.fill(1)
        standard_deviation = np.zeros((3, 3))
        standard_deviation.fill(0.01)

        num_samples = 1000

        random_state = np.random.RandomState(0)
        x = random_state.multivariate_normal(mean, standard_deviation, size=num_samples)
        y = random_state.multivariate_normal(mean, standard_deviation, size=num_samples)

        es = EnergyStatistics(x, y)
        es_test_result = es.get_statistics_and_probabilities(num_tests=100)

        np.testing.assert_almost_equal(es_test_result.e, 0, decimal=3)
        np.testing.assert_almost_equal(es_test_result.e_p, 0, decimal=0)
        np.testing.assert_almost_equal(es_test_result.t, 0.124, decimal=3)
        np.testing.assert_almost_equal(es_test_result.t_p, 0, decimal=0)
        np.testing.assert_almost_equal(es_test_result.h, 0, decimal=3)
        np.testing.assert_almost_equal(es_test_result.h_p, 0, decimal=0)

    def test_get_statistics_and_probabilities_different_distribution_samples(self):
        """Test energy statistics test for clearly different distributions for sufficiently large samples."""
        num_samples = 1000
        random_state = np.random.RandomState(0)

        mean1, standard_deviation1 = 200, 0.1
        mean2, standard_deviation2 = 1, 0.1
        x = random_state.normal(mean1, standard_deviation1, size=num_samples)
        y = random_state.normal(mean2, standard_deviation2, size=num_samples)

        es = EnergyStatistics(x, y)
        es_test_result = es.get_statistics_and_probabilities(num_tests=100)
        np.testing.assert_almost_equal(es_test_result.e, 397.767, decimal=3)
        np.testing.assert_almost_equal(es_test_result.e_p, 0, decimal=0)
        np.testing.assert_almost_equal(es_test_result.t, 198883.754, decimal=3)
        np.testing.assert_almost_equal(es_test_result.t_p, 0, decimal=0)
        np.testing.assert_almost_equal(es_test_result.h, 0.999, decimal=3)
        np.testing.assert_almost_equal(es_test_result.h_p, 0, decimal=0)
