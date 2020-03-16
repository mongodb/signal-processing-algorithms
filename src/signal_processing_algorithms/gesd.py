# -*- coding: utf-8 -*-
"""
GESD based Detect outliers.

Generalized ESD Test for Outliers
see 'GESD<https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm>'
"""
import collections

from typing import List

import numpy as np
import numpy.ma as ma
import structlog

from scipy.stats import t

LOG = structlog.get_logger()

GesdResult = collections.namedtuple(
    "GesdResult",
    ["count", "suspicious_indexes", "test_statistics", "critical_values", "all_z_scores"],
)
"""
A named tuple for the results of the GESD algorithm.

The outliers are in suspicious_indexes[count:].
The low confidence outliers are in suspicious_indexes[:count].

:type count: int,
:type suspicious_indexes: list(int)
:type test_statistics: list(float)
:type critical_values: list(float)
:type all_z_scores: list(float, float)
:type series: list(float)
"""


def gesd(
    data: List[float], max_outliers: int = 10, significance_level: float = 0.05, mad: bool = False
) -> GesdResult:
    """
    Perform a Generalized ESD Test for Outliers.

    The generalized ESD(Extreme Studentized Deviate) test (Rosner 1983) is used to detect one or
    more outliers in a univariate data set that follows an approximately normal distribution.

    Usage:
        gesd_result = gesd(
            series,
            max_outliers,
            significance_level=significance,
            mad=True)

        count = gesd_result.count
        indexes = gesd_result.suspicious_indexes

        print("outliers indexes {}".format(indexes[:count])
        print("low confidence outliers indexes {}".format(indexes[count:])


    If the standard deviation of the series data is zero then the outlier detection will bail out.
    For non-mad this entails a constant series or sub-series so this behaviour makes sense.

    In the MAD case, this may mean that the series is constant or that a majority of the series
    are the median value. The data should be validated to avoid this issue.

    Note: the test_statistics array is signed, this allows determination of the outlier above
    or below the mean.

    :param data: The data to test.
    :param max_outliers: Test for up to max outliers.
    :param significance_level: Test for up to max outliers.
    :param mad: Use Median Absolute Deviation.
    :return: The number of outliers, suspicious indexes, test_statistics, critical_values, z_values.
    see 'here<https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm>'
    """
    if data is None or np.size(data) == 0:
        raise ValueError("No Data ({})".format(data))
    length = len(data)
    if max_outliers < 1:
        raise ValueError(
            "max_outliers({max_outliers}) must be >= 1".format(max_outliers=max_outliers)
        )
    if max_outliers >= len(data):
        raise ValueError(
            "max_outliers({max_outliers}) <= length({length})".format(
                length=len(data) if data is not None else None, max_outliers=max_outliers
            )
        )
    if significance_level <= 0.0 or significance_level >= 1.0:
        raise ValueError(
            "invalid significance_level({significance_level})".format(
                significance_level=significance_level
            )
        )

    series = ma.array(data)
    test_statistics = []
    critical_values = []
    potential_outlier_indices = []

    # max outliers must be less than length, the standard deviation and mad of a single entry list
    # are 0 so z score would be nan.
    m_outliers = min(max_outliers, length - 1)

    indexes = np.arange(m_outliers, dtype=int)
    for i in indexes:
        LOG.debug("iteration", i=i, mad=mad, series=series)
        if mad:
            # sigma in this case is an estimate of .75 quantile * MAD
            # note : 1.4826 == 1 / Q(.75) == 1 / 0.675
            center = np.ma.median(series)
            sigma = 1.4826 * np.ma.median(np.fabs(series - center))
        else:
            center = series.mean()
            sigma = series.std(ddof=1)

        if sigma == 0:
            break

        z_scores = (series - center) / sigma
        if i == 0:
            all_z_scores = (series - center) / sigma

        LOG.debug("calculate", z_scores=np.array([np.arange(length, dtype=int), z_scores]).T)

        max_z_score_index = np.fabs(z_scores).argmax()
        max_z_score = z_scores[max_z_score_index]

        # 2 sided test for significance.
        significance_result = 1.0 - significance_level / (2.0 * (length - i))

        # Percent point function with df (degrees of freedom).
        percentage_point = t.ppf(significance_result, df=length - i - 2)
        value = (
            (length - i - 1)
            * percentage_point
            / np.sqrt((length - i - 2 + percentage_point ** 2) * (length - i))
        )

        # Update results.
        potential_outlier_indices.append(max_z_score_index)
        test_statistics.append(max_z_score)
        critical_values.append(value)

        LOG.debug(
            "iteration complete",
            z_scores=np.array(
                [
                    np.arange(max_outliers, dtype=int),
                    test_statistics,
                    critical_values,
                    np.greater(test_statistics, critical_values),
                ]
            ).T,
        )

        # Mask and exclude the selected value from the next iteration.
        series[max_z_score_index] = ma.masked

    LOG.debug("values calculated", max_z_scores=test_statistics, lambda_values=critical_values)
    if potential_outlier_indices:
        for number_outliers in range(len(potential_outlier_indices), 0, -1):
            if np.abs(test_statistics[number_outliers - 1]) > critical_values[number_outliers - 1]:
                LOG.debug(
                    "outliers discovered",
                    number_outliers=number_outliers,
                    outliers=potential_outlier_indices[0:number_outliers],
                )

                return GesdResult(
                    number_outliers,
                    potential_outlier_indices,
                    test_statistics,
                    critical_values,
                    all_z_scores[potential_outlier_indices],
                )
    return GesdResult(
        0,
        potential_outlier_indices,
        test_statistics,
        critical_values,
        all_z_scores[potential_outlier_indices] if potential_outlier_indices else [],
    )
