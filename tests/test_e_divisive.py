"""
E-Divisive related tests.
"""
from unittest.mock import MagicMock
import numpy as np

from signal_processing_algorithms.e_divisive_native_wrapper import LOADED
from signal_processing_algorithms.e_divisive import EDivisive
import pytest

EMPTY_NP_ARRAY = np.array([], dtype=np.float)


class TestEDivisiveTIG1423(object):
    """
    Test for TIG-1423.
    """

    def test_ctor_defaults(self):
        """
        Test that EDivisive applies defaults for None.
        """
        e_divisive = EDivisive(pvalue=None, permutations=None)
        assert e_divisive.pvalue is not None
        assert e_divisive.permutations is not None


class TestEDivisive(object):
    """
    Test for EDivisive class methods.
    """

    def test_series_none(self):
        """
        Test that compute_change_points parameters are validated.
        """
        e_divisive = EDivisive()
        e_divisive._compute_change_points = MagicMock()

        e_divisive.compute_change_points(None)

        series_arg = e_divisive._compute_change_points.call_args[0][0]
        assert isinstance(series_arg, np.ndarray)
        assert series_arg.size == 0

    def test_series_string(self):
        """
        Test that compute_change_points parameters are validated.
        """
        e_divisive = EDivisive()
        with pytest.raises(ValueError, match=r"could not convert string to float: 'string'"):
            e_divisive.compute_change_points("string")

    def test_series_empty(self):
        """
        Test that compute_change_points parameters are validated.
        """
        e_divisive = EDivisive()
        e_divisive._compute_change_points = MagicMock()

        e_divisive.compute_change_points([])

        series_arg = e_divisive._compute_change_points.call_args[0][0]
        assert isinstance(series_arg, np.ndarray)
        assert series_arg.size == 0

    def test_qhat_values_empty(self):
        """
        Test that qhat_values can accept an empty series.
        """
        assert EDivisive().qhat_values(EMPTY_NP_ARRAY).size == 0

    def test_native_package_included_in_package(self):
        """
        Test that the native implementation of e_divisive is included in the package
        """
        EDivisive()
        assert LOADED
