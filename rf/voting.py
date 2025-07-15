import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Voting(ABC):
    """
    Base class for forrest voting algorithm
    """
    def __init__(self, weights: NDArray | None = None) -> None:
        self.name = self.__class__.__name__
        self.weights = weights if weights is not None else np.ones(1)

    def __call__(self, predictions: NDArray) -> NDArray:
        return self._vote(predictions)

    @abstractmethod
    def _vote(self, predictions: NDArray) -> NDArray:
        pass


class Majority(Voting):
    """
    prediction based on majority results
    """
    def __init__(self, weights: NDArray | None = None) -> None:
        super().__init__(weights)

    def _vote(self, predictions: NDArray) -> NDArray:
        majorityVotes = []
        weightedPredictions = predictions * self.weights
        for i in range(predictions.shape[0]):
            unique, counts = np.unique(weightedPredictions[i, :], return_counts=True)
            majorityVote = unique[np.argmax(counts)]
            majorityVotes.append(majorityVote)
        return np.array(majorityVotes)


class Weighted(Voting):
    """
    prediction based on weighted majority
    """
    def __init__(self, weights: NDArray | None = None) -> None:
        super().__init__(weights)

    def _vote(self, predictions: NDArray) -> NDArray:
        weightedPredictions = self.weights * predictions
        confidenceScores = np.sum(weightedPredictions, axis=1) / np.sum(self.weights)
        return np.round(confidenceScores)


class Average(Voting):
    """
    the average of what the ensamble voted on
    """
    def __init__(self, weights: NDArray | None = None) -> None:
        super().__init__(weights)

    def _vote(self, predictions: NDArray) -> NDArray:
        return np.average(predictions, axis=1, weights=self.weights)


class Median(Voting):
    """
    weighted average voting
    """
    def __init__(self, weights: NDArray | None = None) -> None:
        super().__init__(weights)

    def _vote(self, predictions: NDArray) -> NDArray:
        weightedPredictions = self.weights * predictions
        return np.median(weightedPredictions, axis=1)
