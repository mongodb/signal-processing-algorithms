"""
E-Divisive related tests.
"""
import numpy as np
import pytest

from signal_processing_algorithms.e_divisive import default_implementation

EMPTY_NP_ARRAY = np.array([], dtype=np.float64)


class TestEDivisive(object):
    """
    Test for EDivisive methods.
    """

    def test_ctor_defaults(self):
        """
        Test that EDivisive applies defaults for None.
        """
        e_divisive = default_implementation()
        assert e_divisive._significance_tester._pvalue is not None
        assert e_divisive._significance_tester._permutations is not None

    def test_series_string(self):
        """
        Test that compute_change_points parameters are validated.
        """
        with pytest.raises(ValueError, match=r"could not convert string to float: 'string'"):
            default_implementation().get_change_points("string")

    def test_absolute_certainty(self):
        algo = default_implementation()
        algo._significance_tester._pvalue = 0
        assert algo.fit_predict([0] * 100 + [1] * 100) == [100]

    def test_default_implementation_sanity(self):
        algo = default_implementation()
        for cp in range(4, 100):
            assert algo.fit_predict([0] * cp + [1] * cp) == [cp]
