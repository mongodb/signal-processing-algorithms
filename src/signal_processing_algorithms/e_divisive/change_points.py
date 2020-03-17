"""A change point detected by E-Divisive."""
from typing import Optional


class EDivisiveChangePoint:
    """A change point."""

    __slots__ = ["index", "qhat", "probability"]

    def __init__(self, index: int = 0, qhat: float = 0, probability: Optional[float] = None):
        """
        Create an E-Divisive change point, representing a change point found by E-Divisive algorithm.

        :param index: Index of the change point.
        :param qhat: The Q-Hat metric for the change point.
        :param probability: The probability that the change point is valid, based on a permutation test.
        """
        self.index = index
        self.qhat = qhat
        self.probability = probability

    index: int
    qhat: float
    probability: Optional[float]

    def __eq__(self, other: object) -> bool:
        """
        Check whether one change point is equal to another based on index, qhat, and probability.

        :param other: The other change point.
        :return: True if they are equal, false otherwise.
        """
        if not isinstance(other, EDivisiveChangePoint):
            return False
        return (
            self.index == other.index
            and self.qhat == other.qhat
            and self.probability == other.probability
        )
