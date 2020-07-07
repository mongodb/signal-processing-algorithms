from math import isfinite

import numpy as np

from hypothesis import assume, given
from hypothesis.strategies import floats, integers, lists

from signal_processing_algorithms.e_divisive.calculators import cext_calculator, numpy_calculator


@given(lists(floats(0, 1000, allow_infinity=False, allow_nan=False), min_size=10, max_size=1000))
def test_diff_equivalence(data):
    for x in data:
        assume(isfinite(x))
    series = np.array(data, dtype=float)
    np_res = numpy_calculator.calculate_diffs(series)
    cext_res = cext_calculator.calculate_diffs(series)
    assert np.array_equal(np_res, cext_res)


@given(lists(floats(0, 1000, allow_infinity=False, allow_nan=False), min_size=100, max_size=1000))
def test_q_values(data):
    series = np.array(data, dtype=float)
    diffs = numpy_calculator.calculate_diffs(series)
    np_res = numpy_calculator.calculate_qhat_values(diffs)
    cext_res = cext_calculator.calculate_qhat_values(diffs)
    np.testing.assert_array_almost_equal(np_res, cext_res, 5)


@given(
    floats(0, 1000, allow_infinity=False, allow_nan=False),
    floats(0, 1000, allow_infinity=False, allow_nan=False),
    floats(0, 1000, allow_infinity=False, allow_nan=False),
    integers(0, 1000),
    integers(0, 1000),
)
def test_calculate_q(cross_term, x_term, y_term, x_len, y_len):
    assume(x_len + y_len != 0)
    q1 = numpy_calculator._calculate_q(cross_term, x_term, y_term, x_len, y_len)
    q2 = cext_calculator._calculate_q(cross_term, x_term, y_term, x_len, y_len)
    assert q1 == q2


@given(
    lists(floats(0, 1000, allow_infinity=False, allow_nan=False), min_size=100, max_size=1000),
    integers(0, 100),
    integers(0, 100),
    integers(0, 100),
    integers(0, 100),
)
def test_square_sum(data, row_start, row_end, column_start, column_end):
    assume(row_start <= row_end)
    assume(column_start <= column_end)
    series = np.array(data, dtype=float)
    diffs = cext_calculator.calculate_diffs(series)
    sum1 = cext_calculator._square_sum(diffs, row_start, row_end, column_start, column_end)
    sum2 = np.sum(diffs[row_start:row_end, column_start:column_end])
    np.testing.assert_almost_equal(sum1, sum2, 5)
