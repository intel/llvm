import statistics
from abc import ABC, abstractmethod

from options import options


class Aggregator(ABC):
    """
    Aggregator classes used to "aggregate" a pool of elements, and produce an
    "average" (precisely, some "measure of central tendency") from the elements.
    """

    @staticmethod
    @abstractmethod
    def get_type() -> str:
        """
        Return a string indicating the type of average this aggregator
        produces.
        """
        pass

    @abstractmethod
    def add(self, n: float):
        """
        Add/aggregate an element to the pool of elements used by this aggregator
        to produce an average calculation.
        """
        pass

    @abstractmethod
    def get_avg(self) -> float:
        """
        Produce an average from the pool of elements aggregated using add().
        """
        pass


class SimpleMedian(Aggregator):
    """
    Simple median calculation: if the number of samples being generated are low,
    this is the fastest median method.
    """

    def __init__(self, starting_elements: list = []):
        self.elements = starting_elements

    @staticmethod
    def get_type() -> str:
        return "median"

    def add(self, n: float):
        self.elements.append(n)

    def get_avg(self) -> float:
        return statistics.median(self.elements)


class EWMA(Aggregator):
    """
    Exponentially weighted moving average based on all elements added to the
    aggregator.
    """

    def __init__(self, starting_elements: list = []):
        self.elements = starting_elements

    @staticmethod
    def get_type() -> str:
        return "EWMA"

    def add(self, n: float):
        self.elements.append(n)

    def get_avg(self) -> float:
        if len(self.elements) == 0:
            return None  # No elements collected, cannot provide an average

        alpha = options.EWMA_smoothing_factor
        ewma_t = self.elements[0]
        for x_t in self.elements[1:]:
            ewma_t = alpha * x_t + (1 - alpha) * ewma_t
        return ewma_t
