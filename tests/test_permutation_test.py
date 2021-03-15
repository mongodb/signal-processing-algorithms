import numpy as np

from signal_processing_algorithms.energy_statistics.permutation_test import (
    get_energy_statistics_and_probabilities,
)


class TestPermutationTest(object):
    def test_get_energy_statistics_and_probabilities_same_distribution_samples_univariate(self):
        """Test sufficiently large number samples drawn from the same distribution have almost zero H."""
        mean, standard_deviation = 1, 0.1
        num_samples = 1000

        random_state = np.random.RandomState(0)
        x = random_state.normal(mean, standard_deviation, size=num_samples)
        y = random_state.normal(mean, standard_deviation, size=num_samples)

        es_with_probs = get_energy_statistics_and_probabilities(x, y, permutations=100)

        np.testing.assert_almost_equal(es_with_probs.e, 0, decimal=3)
        np.testing.assert_almost_equal(es_with_probs.e_pvalue, 0, decimal=0)
        np.testing.assert_almost_equal(es_with_probs.t, 0.125, decimal=3)
        np.testing.assert_almost_equal(es_with_probs.t_pvalue, 0, decimal=0)
        np.testing.assert_almost_equal(es_with_probs.h, 0.001, decimal=3)
        np.testing.assert_almost_equal(es_with_probs.h_pvalue, 0, decimal=0)

    def test_get_energy_statistics_and_probabilities_same_distribution_samples_multivariate(self):
        """Test sufficiently large number samples drawn from the same distribution have low statistics values."""
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
        np.testing.assert_almost_equal(es_with_probs.e_pvalue, 0, decimal=0)
        np.testing.assert_almost_equal(es_with_probs.t, 0.124, decimal=3)
        np.testing.assert_almost_equal(es_with_probs.t_pvalue, 0, decimal=0)
        np.testing.assert_almost_equal(es_with_probs.h, 0, decimal=3)
        np.testing.assert_almost_equal(es_with_probs.h_pvalue, 0, decimal=0)

    def test_get_energy_statistics_and_probabilities_different_distribution_samples(self):
        """Test energy statistics test for clearly different distributions for sufficiently large samples."""
        num_samples = 1000
        random_state = np.random.RandomState(0)

        mean1, standard_deviation1 = 200, 0.1
        mean2, standard_deviation2 = 1, 0.1
        x = random_state.normal(mean1, standard_deviation1, size=num_samples)
        y = random_state.normal(mean2, standard_deviation2, size=num_samples)

        es_with_probs = get_energy_statistics_and_probabilities(x, y, permutations=100)
        np.testing.assert_almost_equal(es_with_probs.e, 397.767, decimal=3)
        np.testing.assert_almost_equal(es_with_probs.e_pvalue, 0, decimal=0)
        np.testing.assert_almost_equal(es_with_probs.t, 198883.754, decimal=3)
        np.testing.assert_almost_equal(es_with_probs.t_pvalue, 0, decimal=0)
        np.testing.assert_almost_equal(es_with_probs.h, 0.999, decimal=3)
        np.testing.assert_almost_equal(es_with_probs.h_pvalue, 0, decimal=0)
