import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from machineLearning.data.dataLoader import DataSet


class ImpurityMeasure(ABC):
    """
    Abstract base class for impurity measures, such as Gini and Entropy.
    """
    def __init__(self, numClasses: int | None = None) -> None:
        self.name = self.__class__.__name__
        self.numClasses = numClasses

    def __call__(self, data: DataSet | None = None, *, labels: NDArray | None = None, targets: NDArray | None = None, weights: NDArray | None = None, classWeights: NDArray | None = None) -> float:
        """
        Calls the _loss method to compute the impurity measure value of the input target array.
        """
        return self._loss(targets, weights=weights, classWeights=classWeights)

    @abstractmethod
    def _loss(self, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> float:
        """
        Computes the impurity measure value of the input target array.
        """
        pass


class Gini(ImpurityMeasure):
    """
    A class for computing the Gini impurity measure.
    """
    def __init__(self, numClasses: int | None = None, epsilon: float = 10e-9) -> None:
        super().__init__(numClasses=numClasses)
        self.epsilon = epsilon # stability constant

    def _loss(self, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray| None = None) -> float:
        """
        Computes the Gini impurity measure value of the input target array.
        """
        if self.numClasses is None:
            uniTargets, inv = np.unique(targets, return_inverse=True)  # Find unique classes and indices mapping
        else:
            uniTargets = np.arange(self.numClasses)
            inv = targets

        if weights is None:
            weights = np.ones(len(targets)) / len(targets)  # Uniform weights if none provided

        weightedCounts = np.zeros(len(uniTargets))
        for idx, classLabel in enumerate(uniTargets):
            weightedCounts[idx] = np.sum(weights[inv == idx])  # Sum weights for each class

        if classWeights is None:
            classWeights = np.ones(len(uniTargets))  # Uniform class weights if none provided

        totalWeight = np.sum(weightedCounts)
        p = (weightedCounts * classWeights) / (totalWeight + self.epsilon)  # Compute the weighted probability of each class
        return np.sum(p * (1 - p))  # Compute the Gini impurity


class Entropy(ImpurityMeasure):
    """
    A class for computing the Entropy impurity measure.
    """
    def __init__(self, numClasses: int | None = None, epsilon: float = 10e-9) -> None:
        super().__init__(numClasses=numClasses)
        self.epsilon = epsilon # stability constant

    def _loss(self, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> float:
        """
        Computes the Entropy impurity measure value of the input target array.
        """
        if self.numClasses is None:
            uniTargets, inv = np.unique(targets, return_inverse=True)  # Find unique classes and indices mapping
        else:
            uniTargets = np.arange(self.numClasses)
            inv = targets

        if weights is None:
            weights = np.ones(len(targets)) / len(targets)  # Uniform weights if none provided

        weightedCounts = np.zeros(len(uniTargets))
        for idx, classLabel in enumerate(uniTargets):
            weightedCounts[idx] = np.sum(weights[inv == idx])  # Sum weights for each class

        if classWeights is None:
            classWeights = np.ones(len(uniTargets))  # Uniform class weights if none provided

        totalWeight = np.sum(weightedCounts)
        p = (weightedCounts * classWeights) / (totalWeight + self.epsilon)  # Compute the weighted probability of each class

        # Handle zero probabilities to avoid log2(0) which is undefined
        p = p[p > 0]
        return -np.sum(p * np.log2(p))  # Compute the Entropy


class MAE(ImpurityMeasure):
    """
    Computes the mean absolute error (MAE) of the given targets.
    """
    def __init__(self) -> None:
        super().__init__()

    def _loss(self, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray| None = None) -> float:
        if len(targets) == 0:
            return 0.0
        if weights is None:
            weights = np.ones(len(targets)) / len(targets)
        targetsMean = np.mean(targets)
        weightedAbsErrors = weights * np.abs(targets - targetsMean)
        return np.sum(weightedAbsErrors) / np.sum(weights)


class MSE(ImpurityMeasure):
    """
    Computes the mean squared error (MSE) of the given targets.
    """
    def __init__(self) -> None:
        super().__init__()

    def _loss(self, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray| None = None) -> float:
        if len(targets) == 0:
            return 0.0
        if weights is None:
            weights = np.ones(len(targets)) / len(targets)
        targetsMean = np.mean(targets)
        weightedSquaredErrors = weights * (targets - targetsMean) ** 2
        return np.sum(weightedSquaredErrors) / np.sum(weights)


class ODD(ImpurityMeasure):
    """
    Used for anomaly detection, it's stubb class
    """
    def __init__(self) -> None:
        super().__init__()

    def _loss(self, targets: NDArray, *, weights: NDArray| None = None, classWeights: NDArray| None = None) -> float:
        return np.random.random() # return some default value
