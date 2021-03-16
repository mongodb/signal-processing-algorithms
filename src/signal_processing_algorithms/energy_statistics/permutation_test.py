"""Permutation tests for Energy Statistics."""
from typing import List, Union

import numpy as np

from signal_processing_algorithms.distance import get_distance_matrix
from signal_processing_algorithms.energy_statistics.energy_statistics import (
    EnergyStatisticsWithProbabilities,
    get_energy_statistics,
    _get_energy_statistics_from_distance_matrix,
    _get_valid_input,
)

