"""Functions to help setup deterministic results when using random numbers."""
# E-Divisive's definition requires it to permute change-windows
# which leads to non-determinism: we need to always get the
# same change-point results when running on the same input.
import random

from contextlib import contextmanager
from typing import Generator


@contextmanager
def deterministic_random(seed: float) -> Generator:
    """
    Call random.seed(seed) during invocation and then restore state after.

    :param seed: RNG seed.
    :return: Deterministic random context.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)
