# -*- coding: utf-8 -*-
"""
Unit tests for signal_processing/outliers/gesd.py.
"""
import random

import numpy as np
import pytest

from signal_processing_algorithms.determinism import deterministic_random
from signal_processing_algorithms.gesd import gesd


class TestGesdInputs(object):
    """ Test error handling. """

    def test_no_data(self):
        """Test no data."""
        with pytest.raises(ValueError, match=r"No Data"):
            gesd(None)
        with pytest.raises(ValueError, match=r"No Data"):
            gesd([])

    def test_max_outliers(self):
        """Test insufficient data."""
        with pytest.raises(ValueError, match=r"max_outliers.* must be >= 1"):
            gesd([1], 0)
        with pytest.raises(ValueError, match=r"max_outliers.* must be >= 1"):
            gesd([1], -1)

    def test_insufficient_data(self):
        """Test insufficient data."""
        with pytest.raises(ValueError, match=r"max_outliers.* <= length"):
            gesd([1])
        with pytest.raises(ValueError, match=r"max_outliers.* <= length"):
            gesd([1] * 10)

    def test_significance_level_zero(self):
        """Test invalid significance_level."""
        with pytest.raises(ValueError, match=r"invalid significance_level"):
            gesd([1] * 20, significance_level=0)
        with pytest.raises(ValueError, match=r"invalid significance_level"):
            gesd([1] * 20, significance_level=0)

    def test_significance_level_lt_zero(self):
        """Test invalid significance_level."""
        with pytest.raises(ValueError, match=r"invalid significance_level"):
            gesd([1] * 20, significance_level=-1)

    def test_significance_level_gt_one(self):
        """Test invalid significance_level."""
        with pytest.raises(ValueError, match=r"invalid significance_level"):
            gesd([1] * 20, significance_level=1)


class TestSimple(object):
    """ Test Simple data. """

    def test_flat(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            [1] * 20
        )
        assert 0 == number_outliers
        assert [] == suspicious_indexes
        assert [] == test_statistics
        assert [] == critical_values
        assert [] == all_z_scores

    def test_mad_flat(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            [1] * 20, mad=True
        )
        assert 0 == number_outliers
        assert [] == suspicious_indexes
        assert [] == test_statistics
        assert [] == critical_values
        assert [] == all_z_scores


class TestSimpleData(object):
    """ Test Simple data. """

    series = [-1] * 203 + [-2] + [-1] * 75

    def test_standard(self):
        """Test gesd on almost flat data."""
        number_outliers, suspicious_indexes, _, _, _ = gesd(self.series)
        assert 1 == number_outliers
        assert [203] == suspicious_indexes

    def test_mad(self):
        """Test gesd on almost flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            self.series, mad=True
        )
        assert 0 == number_outliers
        assert [] == suspicious_indexes
        assert [] == test_statistics
        assert [] == critical_values
        assert [] == all_z_scores


class TestReal(object):
    """ Test Real data. """

    def test_standard(self, real_series):
        """Test gesd on real data with standard."""
        number_outliers, suspicious_indexes, _, _, _ = gesd(real_series)
        assert 2 == number_outliers
        assert [90, 4, 27, 88, 37, 64, 126, 78, 74, 47] == suspicious_indexes

    def test_mad(self, mad_series):
        """Test gesd on real data with Median Absolute Deviation."""
        number_outliers, suspicious_indexes, _, _, _ = gesd(mad_series, mad=True)
        assert 2 == number_outliers
        assert [90, 4, 27, 37, 64, 78, 74, 88, 47, 60] == suspicious_indexes


FIRST_OUTLIER = 700
SECOND_OUTLIER = 100
THIRD_OUTLIER = 5
with deterministic_random(3.1415):
    SINGLE = [FIRST_OUTLIER if i == 7 else random.uniform(0, 1) for i in range(15)]
    DOUBLE = SINGLE + [SECOND_OUTLIER if i == 5 else random.uniform(0, 1) for i in range(10)]
    TRIPLE = DOUBLE + [THIRD_OUTLIER if i == 5 else random.uniform(0, 1) for i in range(10)]


class TestMeanOutliers(object):
    """ Test standard z score. """

    # pylint: disable=unused-variable
    def test_single(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            SINGLE, max_outliers=1
        )
        assert 1 == number_outliers
        assert [7] == suspicious_indexes

    def test_single_max_10(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            SINGLE
        )
        assert 1 == number_outliers
        assert 7 == suspicious_indexes[0]

    def test_double(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            DOUBLE, max_outliers=2
        )
        assert 2 == number_outliers
        assert [7, 20] == suspicious_indexes

    def test_double_max_10(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            DOUBLE
        )
        assert 2 == number_outliers
        assert 7 == suspicious_indexes[0]
        assert 20 == suspicious_indexes[1]

    def test_triple(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            TRIPLE, max_outliers=3
        )
        assert 3 == number_outliers
        assert [7, 20, 30] == suspicious_indexes

    def test_triple_max_2(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            TRIPLE, max_outliers=2
        )
        assert 2 == number_outliers
        assert [7, 20] == suspicious_indexes

    def test_triple_max_10(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            TRIPLE
        )
        assert 3 == number_outliers
        assert [7, 20, 30] == suspicious_indexes[:3]


class TestMedianOutlier(object):
    """ Test Median Absolute Deviation. """

    def test_single(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            SINGLE, max_outliers=1, mad=True
        )
        assert 1 == number_outliers
        assert [7] == suspicious_indexes

    def test_single_max_10(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            SINGLE, mad=True
        )
        assert 10 == number_outliers
        assert 7 == suspicious_indexes[0]

    def test_double(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            DOUBLE, max_outliers=2
        )
        assert 2 == number_outliers
        assert [7, 20] == suspicious_indexes

    def test_double_max_10(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            DOUBLE, max_outliers=2, mad=True
        )
        assert 2 == number_outliers
        assert [7, 20] == suspicious_indexes

    def test_triple(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            TRIPLE, max_outliers=3, mad=True
        )
        assert 3 == number_outliers
        assert [7, 20, 30] == suspicious_indexes

    def test_triple_max_2(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            TRIPLE, max_outliers=2, mad=True
        )
        assert 2 == number_outliers
        assert [7, 20] == suspicious_indexes

    def test_triple_max_10(self):
        """Test gesd on flat data."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            TRIPLE, mad=True
        )
        assert 3 == number_outliers
        assert [7, 20, 30] == suspicious_indexes[:3]


class TestCanonical(object):
    """Test canonical example from
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm#Generalized%20ESD%20Test%20Example.

      H0:  there are no outliers in the data
      Ha:  there are up to 10 outliers in the data

      Significance level:  α = 0.05
      Critical region:  Reject H0 if Ri > critical value

      Summary Table for Two-Tailed Test
      ---------------------------------------
            Exact           Test     Critical
        Number of      Statistic    Value, λi
      Outliers, i      Value, Ri          5 %
      ---------------------------------------
              1          3.118          3.158
              2          2.942          3.151
              3          3.179          3.143 *
              4          2.810          3.136
              5          2.815          3.128
              6          2.848          3.120
              7          2.279          3.111
              8          2.310          3.103
              9          2.101          3.094
             10          2.067          3.085
    """

    # pylint: disable=unused-variable

    ROSNER = [
        -0.25,
        0.68,
        0.94,
        1.15,
        1.20,
        1.26,
        1.26,
        1.34,
        1.38,
        1.43,
        1.49,
        1.49,
        1.55,
        1.56,
        1.58,
        1.65,
        1.69,
        1.70,
        1.76,
        1.77,
        1.81,
        1.91,
        1.94,
        1.96,
        1.99,
        2.06,
        2.09,
        2.10,
        2.14,
        2.15,
        2.23,
        2.24,
        2.26,
        2.35,
        2.37,
        2.40,
        2.47,
        2.54,
        2.62,
        2.64,
        2.90,
        2.92,
        2.92,
        2.93,
        3.21,
        3.26,
        3.30,
        3.59,
        3.68,
        4.30,
        4.64,
        5.34,
        5.42,
        6.01,
    ]

    CANONICAL_STATS = [3.118, 2.942, 3.179, 2.810, 2.815, 2.848, 2.279, 2.310, 2.101, 2.067]
    CANONICAL_CRITICAL = [3.158, 3.151, 3.143, 3.136, 3.128, 3.120, 3.111, 3.103, 3.094, 3.085]
    CANONICAL_INDEXES = [53, 52, 51, 50, 0, 49, 48, 47, 1, 46]

    def test_canonical(self):
        """Test gesd implementation."""
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            self.ROSNER
        )

        assert 3 == number_outliers
        assert np.array_equal(suspicious_indexes, self.CANONICAL_INDEXES)

        assert all(np.isclose(self.CANONICAL_STATS, np.fabs(test_statistics), rtol=0.001))
        assert all(np.isclose(self.CANONICAL_CRITICAL, critical_values, rtol=0.001))

    def test_mad(self):
        """ Test MAD z score. """
        number_outliers, suspicious_indexes, test_statistics, critical_values, all_z_scores = gesd(
            self.ROSNER, mad=True
        )

        assert 4 == number_outliers
        assert np.array_equal(
            suspicious_indexes[:number_outliers], self.CANONICAL_INDEXES[:number_outliers]
        )
