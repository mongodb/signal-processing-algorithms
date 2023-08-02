"""C Extension E-Divisive calculator."""
import os

from ctypes import c_bool, c_double, c_int

import numpy as np
import structlog

LOG = structlog.get_logger()

# A flag indicating if the native library was found. Pessimistically set to False.
so_path = os.path.dirname(os.path.abspath(__file__))
LOG.error(f"LOOK HERE: {so_path}")
LOG.error(f"{os.listdir(so_path)}")
try:
    # input type for the cos_doubles function
    # must be a double array, with single dimension that is contiguous
    ARRAY_DOUBLE = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")
    MATRIX_DOUBLE = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="CONTIGUOUS")

    # load the library, using numpy mechanisms
    LIB_E_DIVISIVE = np.ctypeslib.load_library("_e_divisive", so_path)

    # setup the return types and argument types
    LIB_E_DIVISIVE.t_stat_values.restype = c_bool
    LIB_E_DIVISIVE.t_stat_values.argtypes = [MATRIX_DOUBLE, ARRAY_DOUBLE, c_int]

    # setup the return types and argument types
    LIB_E_DIVISIVE.calculate_distance_matrix.restype = c_bool
    LIB_E_DIVISIVE.calculate_distance_matrix.argtypes = [ARRAY_DOUBLE, MATRIX_DOUBLE, c_int]

    # setup the return types and argument types
    LIB_E_DIVISIVE.square_sum.restype = c_double
    LIB_E_DIVISIVE.square_sum.argtypes = [MATRIX_DOUBLE, c_int, c_int, c_int, c_int, c_int]

    # setup the return types and argument types
    LIB_E_DIVISIVE.calculate_t.restype = c_double
    LIB_E_DIVISIVE.calculate_t.argtypes = [c_double, c_double, c_double, c_int, c_int]

    def _calculate_t(
        cross_term: float, x_term: float, y_term: float, x_len: int, y_len: int
    ) -> float:
        result = LIB_E_DIVISIVE.calculate_t(cross_term, x_term, y_term, x_len, y_len)
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
            np.ascontiguousarray(matrix, dtype=np.float64),
            size,
            row_start,
            row_end,
            column_start,
            column_end,
        )

        return result

    def calculate_distance_matrix(series: np.ndarray) -> np.ndarray:
        """
        Return the matrix of pairwise distances within the series.

        :param series: An m x n array of m observations for n variables.
        :return: An m x m array where (i,j)th value is the distance between the observation
        at i-th row of series and j-th row of series.
        """
        if not isinstance(series, np.ndarray):
            series = np.array(series, dtype=float)
        size = len(series)
        distance_matrix = np.zeros((size, size), dtype=np.float64)
        result = LIB_E_DIVISIVE.calculate_distance_matrix(series, distance_matrix, size)
        if result is not True:
            raise Exception("Native E-Divisive returned unexpected value {}".format(result))

        return distance_matrix

    def calculate_t_values(distance_matrix: np.ndarray) -> np.ndarray:
        """
        Marshall the parameters and call the native t_stat_values function.

        :param distance_matrix: The distance matrix.
        :return: The calculated t values.
        """
        size = len(distance_matrix)
        t_stat_values = np.zeros(len(distance_matrix))
        result = LIB_E_DIVISIVE.t_stat_values(
            np.ascontiguousarray(distance_matrix, dtype=np.float64), t_stat_values, size
        )
        if result is not True:
            raise Exception("Native E-Divisive returned unexpected value {}".format(result))

        return t_stat_values

    C_EXTENSION_LOADED = True
except OSError:
    C_EXTENSION_LOADED = False
    LOG.warn("native E-Divisive could not be loaded", loaded=False, so_path=so_path, exc_info=1)
