from math import isfinite

import numpy as np

from hypothesis import assume, given
from hypothesis.strategies import floats, integers, lists

from signal_processing_algorithms.energy_statistics import cext_calculator, energy_statistics


@given(lists(floats(0, 1000, allow_infinity=False, allow_nan=False), min_size=10, max_size=1000))
def test_diff_equivalence(data):
    for x in data:
        assume(isfinite(x))
    series = np.array(data, dtype=float)
    series = np.atleast_2d(series).T
    np_res = energy_statistics._get_distance_matrix(series, use_c_if_possible=False)
    cext_res = energy_statistics._get_distance_matrix(series, use_c_if_possible=True)
    assert np.array_equal(np_res, cext_res)


@given(lists(floats(0, 1000, allow_infinity=False, allow_nan=False), min_size=100, max_size=1000))
def test_t_values(data):
    series = np.array(data, dtype=float)
    series = np.atleast_2d(series).T
    diffs = energy_statistics._get_distance_matrix(series, use_c_if_possible=True)
    np_res = energy_statistics._calculate_t_stats(diffs, use_c_if_possible=False)
    cext_res = cext_calculator.calculate_t_values(diffs)
    np.testing.assert_array_almost_equal(np_res, cext_res, 5)


@given(
    floats(0, 1000, allow_infinity=False, allow_nan=False),
    floats(0, 1000, allow_infinity=False, allow_nan=False),
    floats(0, 1000, allow_infinity=False, allow_nan=False),
    integers(0, 1000),
    integers(0, 1000),
)
def test_calculate_t(cross_term, x_term, y_term, x_len, y_len):
    assume(x_len + y_len != 0)
    t1 = energy_statistics._calculate_stats(x_term, y_term, cross_term, x_len, y_len)[1]
    t2 = cext_calculator._calculate_t(cross_term, x_term, y_term, x_len, y_len)
    np.testing.assert_almost_equal(t1, t2, 5)


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
    diffs = cext_calculator.calculate_distance_matrix(series)
    sum1 = cext_calculator._square_sum(diffs, row_start, row_end, column_start, column_end)
    sum2 = np.sum(diffs[row_start:row_end, column_start:column_end])
    np.testing.assert_almost_equal(sum1, sum2, 5)
