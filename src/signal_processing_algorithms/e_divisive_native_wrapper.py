"""A wrapper for E-Divisive's C implementation using numpy's ctypeslib."""

from ctypes import c_int

import numpy as np
import numpy.ctypeslib as npct
import os
import structlog

LOG = structlog.get_logger()

# A flag indicating if the native library was found. Pessimistically set to False.
LOADED = False
so_path = os.path.dirname(os.path.abspath(__file__))
try:
    # input type for the cos_doubles function
    # must be a double array, with single dimension that is contiguous
    ARRAY_DOUBLE = npct.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")
    MATRIX_DOUBLE = npct.ndpointer(dtype=np.double, ndim=2, flags="CONTIGUOUS")

    # load the library, using numpy mechanisms
    LIB_E_DIVISIVE = npct.load_library("_e_divisive", so_path)

    # setup the return types and argument types
    LIB_E_DIVISIVE.qhat_values.restype = c_int
    LIB_E_DIVISIVE.qhat_values.argtypes = [ARRAY_DOUBLE, MATRIX_DOUBLE, ARRAY_DOUBLE, c_int]

    # setup the return types and argument types
    LIB_E_DIVISIVE.calculate_diffs.restype = c_int
    LIB_E_DIVISIVE.calculate_diffs.argtypes = [ARRAY_DOUBLE, MATRIX_DOUBLE, c_int]

    def qhat_values_wrapper(
        series: np.ndarray, diffs: np.ndarray, qhat_values: np.ndarray
    ) -> np.ndarray:
        """
        Marshall the parameters and call the native qhat_values function.

        :param np.ndarray series: The series data.
        :param np.2darray(float) diffs: The diffs matrix.
        :param np.ndarray qhat_values: The array to store the qhat values.
        :return: The calculated qhat values.
        :rtype: np.ndarray.
        :raises: Exception if the native function doesn't return 0.
        """
        if not isinstance(series, np.ndarray):
            series = np.array(series, dtype=float)
        size = len(series)
        result = LIB_E_DIVISIVE.qhat_values(series, diffs, qhat_values, size)
        if result != 0:
            raise Exception("Native E-Divisive returned unexpected value {}".format(result))

        return qhat_values

    def qhat_diffs_wrapper(series: np.ndarray) -> np.ndarray:
        """
        Marshall the parameters and call the native qhat_values function.

        :param np.ndarray series: The series data.
        :param np.2darray(float) diffs: The diffs matrix.
        :param np.ndarray qhat_values: The array to store the qhat values.
        :return: The calculated qhat values.
        :rtype: np.ndarray.
        :raises: Exception if the native function doesn't return 0.
        """
        if not isinstance(series, np.ndarray):
            series = np.array(series, dtype=float)
        size = len(series)
        diffs = np.zeros((size, size), dtype=np.float)
        result = LIB_E_DIVISIVE.calculate_diffs(series, diffs, size)
        if result != 0:
            raise Exception("Native E-Divisive returned unexpected value {}".format(result))

        return diffs

    LOADED = True
except Exception:
    LOG.warn(
        "native E-Divisive could not be loaded, falling back to python implementation",
        loaded=False,
        so_path=so_path,
        exc_info=1,
    )
