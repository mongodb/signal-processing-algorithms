"""Computes the E-Divisive means change points."""

import os
import numpy as np
import structlog

from signal_processing_algorithms import e_divisive_native_wrapper
from signal_processing_algorithms.e_divisive_numpy import EDivisiveNumpyImplementation, QHatValues

LOG = structlog.get_logger()


DSI_DISABLE_NATIVE_E_DIVISIVE = os.environ.get(
    "DSI_DISABLE_NATIVE_E_DIVISIVE", "false"
).lower() in ["true", "t"]


class EDivisive(EDivisiveNumpyImplementation):
    """E-Divisive class that uses C Extensions where available unless disabled."""

    def _calculate_qhat_values(
        self, series: np.ndarray, diffs: np.ndarray, qhat_values: np.ndarray
    ) -> QHatValues:
        if not DSI_DISABLE_NATIVE_E_DIVISIVE and e_divisive_native_wrapper.LOADED:
            # used as the window size in extract_q
            diffs = e_divisive_native_wrapper.qhat_diffs_wrapper(series)

            average_value = np.average(series)
            average_diff = np.average(diffs)
            e_divisive_native_wrapper.qhat_values_wrapper(series, diffs, qhat_values)
            return QHatValues(qhat_values, average_value, average_diff, len(series))
        else:
            if not e_divisive_native_wrapper.LOADED:
                LOG.warn(
                    "falling back to numpy optimized E-Divisive",
                    loaded=e_divisive_native_wrapper.LOADED,
                    DSI_DISABLE_NATIVE_E_DIVISIVE=DSI_DISABLE_NATIVE_E_DIVISIVE,
                )
            else:
                LOG.info(
                    "falling back to numpy optimized E-Divisive",
                    loaded=e_divisive_native_wrapper.LOADED,
                    DSI_DISABLE_NATIVE_E_DIVISIVE=DSI_DISABLE_NATIVE_E_DIVISIVE,
                )
            return super(EDivisive, self)._calculate_qhat_values(series, diffs, qhat_values)
