"""C Extension E-Divisive calculator."""
import os

from ctypes import c_bool, c_double, c_int

import numpy as np
import structlog

LOG = structlog.get_logger()

# A flag indicating if the native library was found. Pessimistically set to False.
so_path = os.path.dirname(os.path.abspath(__file__))
try:
    # input type for the cos_doubles function
    # must be a double array, with single dimension that is contiguous
    ARRAY_DOUBLE = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")
    MATRIX_DOUBLE = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="CONTIGUOUS")

    # load the library, using numpy mechanisms
    LIB_E_DIVISIVE = np.ctypeslib.load_library("_e_divisive", so_path)

    # setup the return types and argument types
    LIB_E_DIVISIVE.qhat_values.restype = c_bool
    LIB_E_DIVISIVE.qhat_values.argtypes = [MATRIX_DOUBLE, ARRAY_DOUBLE, c_int]

    # setup the return types and argument types
    LIB_E_DIVISIVE.calculate_diffs.restype = c_bool
    LIB_E_DIVISIVE.calculate_diffs.argtypes = [ARRAY_DOUBLE, MATRIX_DOUBLE, c_int]

    # setup the return types and argument types
    LIB_E_DIVISIVE.square_sum.restype = c_double
    LIB_E_DIVISIVE.square_sum.argtypes = [MATRIX_DOUBLE, c_int, c_int, c_int, c_int, c_int]

    # setup the return types and argument types
    LIB_E_DIVISIVE.calculate_q.restype = c_double
    LIB_E_DIVISIVE.calculate_q.argtypes = [c_double, c_double, c_double, c_int, c_int]

    def _calculate_q(
        cross_term: float, x_term: float, y_term: float, x_len: int, y_len: int
    ) -> float:
        result = LIB_E_DIVISIVE.calculate_q(cross_term, x_term, y_term, x_len, y_len)
        return result

    def _square_sum(
        matrix: np.ndarray, row_start: int, row_end: int, column_start: int, column_end: int
    ) -> float:
        """
        Calculate the sum of elements in an NxN matrix bounded by [row_start, row_end)x[column_start, column_end).

        :param column_end: Ending of the columns.
        :param column_start: Beginning of the columns.
        :param row_end: Ending of the rows.
        :param row_start: Beginning of the columns.
        :param matrix: The matrix.
        :return: The sum of values.
        """
        size = len(matrix)
        result = LIB_E_DIVISIVE.square_sum(
            np.ascontiguousarray(matrix, dtype=np.float),
            size,
            row_start,
            row_end,
            column_start,
            column_end,
        )

        return result

    def calculate_diffs(series: np.ndarray) -> np.ndarray:
        """
        Marshall the parameters and call the native qhat_values function.

        :param np.ndarray series: The series data.
        :return: The calculated qhat values.
        """
        if not isinstance(series, np.ndarray):
            series = np.array(series, dtype=float)
        size = len(series)
        diffs = np.zeros((size, size), dtype=np.float)
        result = LIB_E_DIVISIVE.calculate_diffs(series, diffs, size)
        if result is not True:
            raise Exception("Native E-Divisive returned unexpected value {}".format(result))

        return diffs

    def calculate_qhat_values(diffs: np.ndarray) -> np.ndarray:
        """
        Marshall the parameters and call the native qhat_values function.

        :param diffs: The diffs matrix.
        :return: The calculated qhat values.
        """
        size = len(diffs)
        qhat_values = np.zeros(len(diffs))
        result = LIB_E_DIVISIVE.qhat_values(
            np.ascontiguousarray(diffs, dtype=np.float), qhat_values, size
        )
        if result is not True:
            raise Exception("Native E-Divisive returned unexpected value {}".format(result))

        return qhat_values

    C_EXTENSION_LOADED = True
except OSError:
    C_EXTENSION_LOADED = False
    LOG.warn("native E-Divisive could not be loaded", loaded=False, so_path=so_path, exc_info=1)
