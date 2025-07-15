import numpy as np
from abc import ABC, abstractmethod


class Error(ABC):
    """
    Class representing a generic error in the context of a Self-Organizing Map (SOM).
    """
    def __init__(self):
        self.name = self.__class__.__name__

    def __call__(self, *args) -> float:
        return self._calcError(*args)

    @abstractmethod
    def _calcError(self, *args) -> float:
        pass


class Topological(Error):
    """
    Class representing a Topological error in a Self-Organizing Map (SOM). This error is computed
    based on the topological structure of the SOM.
    """
    def __init__(self):
        super().__init__()

    def _calcError(self, bestMatchingIndices: np.ndarray, secondMatchingIndices: np.ndarray) -> float:
        """
        Compute the Topological error. The error is calculated as the mean distance between
        the best matching indices and the second best matching indices, minus one.
        """
        distance = np.sum(np.abs(bestMatchingIndices - secondMatchingIndices), axis=-1) - 1
        return np.mean(distance)


class Quantazation(Error):
    """
    Class representing a Quantization error in a Self-Organizing Map (SOM). This error is computed
    based on the discrepancy between the best matching units and the original data.
    """
    def __init__(self):
        super().__init__()

    def _calcError(self, bestMatchingUnits: np.ndarray, dataBatch: np.ndarray) -> float:
        """
        Compute the Quantization error. The error is calculated as the mean Euclidean distance
        between the best matching units and the original data batch.
        """
        distance = np.linalg.norm(bestMatchingUnits - dataBatch, axis=-1)
        return np.mean(distance)