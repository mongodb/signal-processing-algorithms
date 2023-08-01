"""
Unit tests for signal_processing/energy_statistics.py.
"""

import numpy as np
import pytest

from signal_processing_algorithms.determinism import deterministic_numpy_random
from signal_processing_algorithms.energy_statistics.energy_statistics import (
    get_energy_statistics,
    get_energy_statistics_and_probabilities,
)


class TestEnergyStatistics(object):
    """Test for Energy Statistics."""

    def test_no_data(self):
        """Test no data."""
        with pytest.raises(ValueError, match=r"Distribution cannot be empty."):
            get_energy_statistics(None, None)
        with pytest.raises(ValueError, match=r"Distribution cannot be empty."):
            get_energy_statistics([], [])

    def test_invalid_number_of_variables(self):
        """Test invalid number of variables for different observations in the two distributions."""
        with pytest.raises(
            ValueError,
            match=r"all the input array dimensions except for the concatenation axis must match "
            r"exactly, but along dimension 1, the array at index 0 has size 1 and the array at "
            r"index 1 has size 2",
        ):
            get_energy_statistics([1, 2, 3], [[1, 2], [3, 4]])

        with pytest.raises(
            ValueError,
            match=r"all the input array dimensions except for the concatenation axis must match "
            r"exactly, but along dimension 1, the array at index 0 has size 1 and the array at "
            r"index 1 has size 2",
        ):
            get_energy_statistics([[1], [2], [3]], [[1, 2]])

    def test_energy_statistics_same_distribution(self):
        """Test energy statistics for same distribution."""
        x = np.arange(10, 70)
        energy_statistics = get_energy_statistics(x, x)
        assert energy_statistics.e == 0
        assert energy_statistics.t == 0
        assert energy_statistics.h == 0

    def test_energy_statistics_similar_distributions(self):
        """Test energy statistics for similar distributions gives small e,t,h values."""
        x = np.arange(1, 100, 1)
        y = np.arange(1, 105, 1)
        energy_statistics = get_energy_statistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 0.160, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 8.136, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.002, decimal=3)

    def test_energy_statistics_different_distributions(self):
        """Test energy statistics for different distributions gives large e,t h values."""
        x = np.arange(1, 100, 1)
        y = np.arange(10000, 13000, 14)
        energy_statistics = get_energy_statistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 21859.691, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 1481794.709, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.954, decimal=3)

    def test_energy_statistics_univariate_ndarray(self):
        """Test energy statistics for uni-variate distributions."""
        x = np.arange(1, 300)
        y = np.arange(1000, 5000, 10)
        energy_statistics = get_energy_statistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 4257.009, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 728381.015, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.748, decimal=3)

    def test_energy_statistics_univariate_list(self):
        """Test energy statistics for uni-variate distributions."""
        x = [i for i in range(300, 500, 1)]
        y = [i for i in range(400, 900, 11)]
        energy_statistics = get_energy_statistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 268.352, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 10035.943, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.532, decimal=3)

    def test_energy_statistics_multivariate_ndarray(self):
        """Test energy statistics for multivariate distributions."""
        x = [[i, i + 7, i + 4] for i in range(100, 200)]
        y = [[i, i + 2, i + 3] for i in range(100, 400, 12)]
        x = np.asarray(x)
        y = np.asarray(y)
        energy_statistics = get_energy_statistics(x, y)
        np.testing.assert_almost_equal(energy_statistics.e, 137.708, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.t, 2754.172, decimal=3)
        np.testing.assert_almost_equal(energy_statistics.h, 0.373, decimal=3)

    def test_energy_statistics_multivariate_list(self):
        """Test energy statistics for multivariate distributions."""
        x = [[i, i + 1] for i in range(0, 100)]
        y = [[i, i + 10] for i in range(0, 1000)]
        energy_statistics = get_energy_statistics(x, y)
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

        diff_es = get_energy_statistics(x, y)
        similar_es = get_energy_statistics(y, z)

        assert diff_es.e > similar_es.e
        assert diff_es.t > similar_es.t
        assert diff_es.h > similar_es.h

    def test_get_energy_statistics_and_probabilities_same_distribution(self):
        """Test energy statistics and probabilities for the same distribution.

        The results should have large p values and small statistics values."""
        with deterministic_numpy_random(1234):
            x = np.arange(10, 70)
            es_with_probs = get_energy_statistics_and_probabilities(x, x, permutations=100)
            np.testing.assert_almost_equal(es_with_probs.e, 0, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.e_pvalue, 0.990, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t, 0, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t_pvalue, 0.990, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h, 0, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h_pvalue, 0.990, decimal=3)

    def test_get_energy_statistics_and_probabilities_very_similar_distributions(self):
        """Test energy statistics and probabilities for very similar distributions.

        The results should have large p values and small statistics values."""
        with deterministic_numpy_random(1234):
            x = np.arange(10, 70)
            y = np.arange(12, 65)
            es_with_probs = get_energy_statistics_and_probabilities(x, y, permutations=100)
            print(es_with_probs)

            np.testing.assert_almost_equal(es_with_probs.e, 0.211, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.e_pvalue, 0.881, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t, 5.961, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t_pvalue, 0.881, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h, 0.005, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h_pvalue, 0.881, decimal=3)

    def test_get_energy_statistics_and_probabilities_similar_distribution_samples(self):
        """Test sufficiently large number samples drawn from the same distribution causes sampling error.

        P value appears to be low even though the samples are drawn from the same distribution."""
        with deterministic_numpy_random(1234):
            mean, standard_deviation = 1, 0.1
            num_samples = 1000

            random_state = np.random.RandomState(0)
            x = random_state.normal(mean, standard_deviation, size=num_samples)
            y = random_state.normal(mean, standard_deviation, size=num_samples)

            es_with_probs = get_energy_statistics_and_probabilities(x, y, permutations=100)

            np.testing.assert_almost_equal(es_with_probs.e, 0, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.e_pvalue, 0.257, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t, 0.125, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t_pvalue, 0.257, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h, 0.001, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h_pvalue, 0.257, decimal=3)

    def test_get_energy_statistics_and_probabilities_similar_distribution_samples_multivariate(
        self,
    ):
        """Test sufficiently large number samples drawn from the same distribution causes sampling error.

        P value appears to be low even though the samples are drawn from the same distribution."""
        with deterministic_numpy_random(1234):
            mean = np.zeros(3)
            mean.fill(1)
            standard_deviation = np.zeros((3, 3))
            standard_deviation.fill(0.01)

            num_samples = 1000

            random_state = np.random.RandomState(0)
            x = random_state.multivariate_normal(mean, standard_deviation, size=num_samples)
            y = random_state.multivariate_normal(mean, standard_deviation, size=num_samples)

            es_with_probs = get_energy_statistics_and_probabilities(x, y, permutations=100)

            np.testing.assert_almost_equal(es_with_probs.e, 0, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.e_pvalue, 0.643, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t, 0.124, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t_pvalue, 0.643, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h, 0, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h_pvalue, 0.643, decimal=3)

    def test_get_energy_statistics_and_probabilities_different_distribution_samples(self):
        """Test energy statistics test for clearly different distributions for sufficiently large samples.

        Test if results have large statistic values and low p values."""

        with deterministic_numpy_random(1234):
            num_samples = 1000
            random_state = np.random.RandomState(0)

            mean1, standard_deviation1 = 200, 0.1
            mean2, standard_deviation2 = 1, 0.1
            x = random_state.normal(mean1, standard_deviation1, size=num_samples)
            y = random_state.normal(mean2, standard_deviation2, size=num_samples)

            es_with_probs = get_energy_statistics_and_probabilities(x, y, permutations=100)
            np.testing.assert_almost_equal(es_with_probs.e, 397.767, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.e_pvalue, 0.0, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t, 198883.754, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.t_pvalue, 0.0, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h, 0.999, decimal=3)
            np.testing.assert_almost_equal(es_with_probs.h_pvalue, 0.0, decimal=3)
