"""
Unit tests for signal_processing/homogeneity.py.
"""
import numpy as np
import pytest

from signal_processing_algorithms.homogeneity import HomogeneityCalculator


class TestHomogeneity(object):
    """
    Test for homogeneity functions.
    """

    homogeneity_calculator = HomogeneityCalculator()

    def test_no_data(self):
        """Test no data."""
        with pytest.raises(ValueError, match=r"Distribution cannot be empty."):
            self.homogeneity_calculator.get_inhomogeneity(None, None)
        with pytest.raises(ValueError, match=r"Distribution cannot be empty."):
            self.homogeneity_calculator.get_inhomogeneity([], [])

    def test_get_valid_type(self):
        """Test valid type is returned."""
        x = [i for i in range(10)]
        actual = self.homogeneity_calculator.get_valid_type(x)
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.float64

    def test_homogeneity_inhomogeneity_relation(self):
        """Test homogeneity and inhomogeneity coefficients add upto 1."""
        x = np.arange(1, 100)
        y = np.arange(1000, 50000, 100)
        assert (
            self.homogeneity_calculator.get_homogeneity(x, y)
            + self.homogeneity_calculator.get_inhomogeneity(x, y)
            == 1
        )

    def test_same_distribution_homogeneity_long_series(self, long_series):
        """Test homogeneity of a series with respect to itself."""
        assert self.homogeneity_calculator.get_inhomogeneity(long_series, long_series) == 0
        assert self.homogeneity_calculator.get_homogeneity(long_series, long_series) == 1

    def test_same_distribution_homogeneity_real_series(self, real_series):
        """Test homogeneity of a series with respect to itself."""
        assert self.homogeneity_calculator.get_inhomogeneity(real_series, real_series) == 0
        assert self.homogeneity_calculator.get_homogeneity(real_series, real_series) == 1

    def test_same_distribution_homogeneity_robust_series(self, robust_series):
        """Test homogeneity of a series with respect to itself."""
        assert self.homogeneity_calculator.get_inhomogeneity(robust_series, robust_series) == 0
        assert self.homogeneity_calculator.get_homogeneity(robust_series, robust_series) == 1

    def test_different_distributions_homogeneity(self):
        """Test homogeneity."""
        x = np.arange(1, 300)
        y = np.arange(1000, 5000, 10)
        np.testing.assert_almost_equal(
            self.homogeneity_calculator.get_homogeneity(x, y), 0.251, decimal=3
        )

    def test_similar_distributions_homogeneity(self):
        """Test homogeneity."""
        x = np.arange(100, 300, 2)
        y = np.arange(100, 280, 2)
        np.testing.assert_almost_equal(
            self.homogeneity_calculator.get_homogeneity(x, y), 0.989, decimal=3
        )

    def test_same_distribution_increasing_samples_increases_homogeneity(self):
        """Test if samples drawn from the same distribution have higher homogeneity
        on increased sampling for sufficiently large samples."""
        mean, standard_deviation = 1, 0.1
        num_samples_1 = 100
        num_samples_2 = 1000

        random_state = np.random.RandomState(0)
        x = random_state.normal(mean, standard_deviation, size=num_samples_1)
        y = random_state.normal(mean, standard_deviation, size=num_samples_1)

        lower_samples_homogeneity = self.homogeneity_calculator.get_homogeneity(x, y)

        x = random_state.normal(mean, standard_deviation, size=num_samples_2)
        y = random_state.normal(mean, standard_deviation, size=num_samples_2)

        higher_samples_homogeneity = self.homogeneity_calculator.get_homogeneity(x, y)

        assert higher_samples_homogeneity >= lower_samples_homogeneity

    def test_homogeneity_for_same_and_different_distributions(
        self,
    ):
        """Test homogeneity is higher for samples from same distribution
        when compared to samples from different distributions for sufficiently large samples."""

        num_samples = 1000
        random_state = np.random.RandomState(0)

        expectation = 30
        x = random_state.poisson(expectation, size=num_samples)

        mean, standard_deviation = 1, 0.1
        y = random_state.normal(mean, standard_deviation, size=num_samples)
        z = random_state.normal(mean, standard_deviation, size=num_samples)

        different_distributions_homogeneity = self.homogeneity_calculator.get_homogeneity(x, y)
        same_distribution_homogeneity = self.homogeneity_calculator.get_homogeneity(y, z)

        assert different_distributions_homogeneity <= same_distribution_homogeneity
