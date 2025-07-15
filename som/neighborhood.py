import numpy as np
from abc import ABC, abstractmethod


class NeighborhoodFunction(ABC):
    """
    Abstract base class for neighborhood functions, such as Gaussian, Bubble, and Mexican Hat.
    """
    def __init__(self, scale: float) -> None:
        self.name = self.__class__.__name__
        self.scale = scale

    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """
        Calls the _compute method to compute the value of the neighborhood function for the input distance and radius.
        """
        return self._compute(distance)

    @abstractmethod
    def _compute(self, distance: np.ndarray) -> np.ndarray:
        """
        Computes the value of the neighborhood function for the input distance and radius.
        """
        pass


class GuassianNeighborhood(NeighborhoodFunction):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def _compute(self, distance: np.ndarray) -> np.ndarray:
        """
        Computes the value of a Gaussian neighborhood function for a given distance and radius.
        """
        return np.exp(-distance ** 2 / (2 * self.scale ** 2))


class BubbleNeighborhood(NeighborhoodFunction):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def _compute(self, distance: np.ndarray) -> np.ndarray:
        """
        Computes the value of a Bubble neighborhood function for a given distance and radius.
        """
        return np.where(distance <= self.scale, 1, 0)


class MexicanHatNeighborhood(NeighborhoodFunction):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def _compute(self, distance: np.ndarray) -> np.ndarray:
        """
        Computes the value of a Mexican hat neighborhood function for a given distance and radius.
        """
        return (1 - (distance / self.scale) ** 2) * np.exp(-distance ** 2 / (2 * self.scale ** 2))


class LinearNeighborhood(NeighborhoodFunction):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def _compute(self, distance: np.ndarray) -> np.ndarray:
        """
        Computes the value of a linear neighborhood function for a given distance and radius.
        """
        return np.where(distance <= self.scale, 1 - distance / self.scale, 0)


class CosineNeighborhood(NeighborhoodFunction):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def _compute(self, distance: np.ndarray) -> np.ndarray:
        """
        Computes the value of a cosine neighborhood function for a given distance and radius.
        """
        return np.cos(np.pi * distance / self.scale)


class CauchyNeighborhood(NeighborhoodFunction):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def _compute(self, distance: np.ndarray) -> np.ndarray:
        """
        Computes the value of a Cauchy neighborhood function for a given distance and radius.
        """
        return 1 / (1 + (distance / self.scale) ** 2)


class EpanechnikovNeighborhood(NeighborhoodFunction):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def _compute(self, distance: np.ndarray) -> np.ndarray:
        """
        Computes the value of an Epanechnikov neighborhood function for a given distance and radius.
        """
        return np.where(distance <= self.scale, 1 - (distance / self.scale) ** 2, 0)