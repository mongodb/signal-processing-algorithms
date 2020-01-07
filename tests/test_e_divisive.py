"""
E-Divisive related tests.
"""
import numpy as np

from signal_processing_algorithms.e_divisive import default_implementation
from signal_processing_algorithms.e_divisive.calculators.cext_calculator import C_EXTENSION_LOADED
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
        e_divisive = default_implementation()
        assert e_divisive._significance_tester._pvalue is not None
        assert e_divisive._significance_tester._permutations is not None


class TestEDivisive(object):
    """
    Test for EDivisive class methods.
    """

    def test_series_string(self):
        """
        Test that compute_change_points parameters are validated.
        """
        with pytest.raises(ValueError, match=r"could not convert string to float: 'string'"):
            default_implementation().get_change_points("string")

    def test_native_package_included_in_package(self):
        """
        Test that the native implementation of e_divisive is included in the package
        """
        assert C_EXTENSION_LOADED
