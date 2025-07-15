import numpy as np
from abc import ABC, abstractmethod

from numpy.typing import NDArray
from .node import Node


class LeafFunction(ABC):
    """
    Base class leaf value assignment functions
    """
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def __call__(self, node: Node | NDArray) -> float | int | NDArray:
        """
        calls '_leafFunc', makes working with this class easier
        """
        if isinstance(node, Node):
            return self._leafFunc(rawPrediction=node._rawValues)

        return self._leafFunc(rawPrediction=node)

    @abstractmethod
    def _leafFunc(self, rawPrediction: NDArray) -> float | int | NDArray:
        """
        needs to be implemented with every daughter class
        """
        pass

    def _raw(self, node: Node) -> NDArray | Node | None:
        if isinstance(node, Node):
            return node._rawValues

        return node


class Mode(LeafFunction):
    """
    This leaf function is used for classification tasks.
    """
    def __init__(self) -> None:
        super().__init__()

    def _leafFunc(self, rawPrediction: NDArray) -> int:
        # If the rawPrediction is one-hot encoded, convert it to categorical
        if len(rawPrediction.shape) == 2 and rawPrediction.shape[1] > 1:
            targetsCategorical = np.argmax(rawPrediction, axis=1)
        else:
            targetsCategorical = rawPrediction

        # Use np.bincount to get the mode
        mode = np.argmax(np.bincount(targetsCategorical.astype(int)))

        return int(mode)


class Mean(LeafFunction):
    """
    this leaf function is used for regression tasks
    """
    def __init__(self) -> None:
        super().__init__()

    def _leafFunc(self, rawPrediction: NDArray) -> float:
        return float(np.mean(rawPrediction))


class Median(LeafFunction):
    """
    this leaf function is used for regression tasks
    """
    def __init__(self) -> None:
        super().__init__()

    def _leafFunc(self, rawPrediction: NDArray) -> float:
        return float(np.median(rawPrediction))


class Probabilities(LeafFunction):
    """
    This leaf function is used for classification tasks where you want
    probabilities for each class instead of predicting a single class.
    """
    def __init__(self, numClasses: int) -> None:
        super().__init__()
        self.numClasses = numClasses

    def _leafFunc(self, rawPrediction: NDArray) -> NDArray:
        # If rawPrediction is one-hot encoded, convert it to categorical
        if len(rawPrediction.shape) == 2 and rawPrediction.shape[1] > 1:
            targetsCategorical = np.argmax(rawPrediction, axis=1)
        else:
            targetsCategorical = rawPrediction

        # Initialize a vector of zeros with length equal to the number of classes
        classProbs = np.zeros(self.numClasses)

        # Use advanced integer indexing to fill the appropriate cell
        classProbs[targetsCategorical.astype(int)] = 1

        # Normalize to turn counts into probabilities
        classProbs /= np.sum(classProbs)

        return classProbs


class Confidence(LeafFunction):
    """
    This leaf function is used for classification tasks where you want
    the confidence of the mode class.
    """
    def __init__(self) -> None:
        super().__init__()

    def _leafFunc(self, rawPrediction: NDArray) -> NDArray:
        # Convert one-hot encoded rawPrediction to categorical if applicable
        if len(rawPrediction.shape) == 2 and rawPrediction.shape[1] > 1:
            targetsCategorical = np.argmax(rawPrediction, axis=1)
        else:
            targetsCategorical = rawPrediction.astype(int)

        # Compute mode and its count
        unique, counts = np.unique(targetsCategorical, return_counts=True)
        modeCounts = counts.max()
        modeIndex = np.argmax(counts)
        modeValue = unique[modeIndex]

        # Calculate confidence
        confidence = modeCounts / len(targetsCategorical)

        return np.array([modeValue, modeCounts, confidence])


class AnomalyDetection(LeafFunction):
    """
    Leaf function for anomaly detections using ODD trees.
    """
    def __init__(self, strategy: str = 'level') -> None:
        super().__init__()
        self.strategy = strategy  # Determines which node attribute to use (e.g., 'level')

    def __call__(self, node: Node | NDArray) -> NDArray:
        return self._leafFunc(node=node)

    def _leafFunc(self, node: Node) -> float | int | NDArray:
        # Use self.strategy to access the corresponding attribute from the node
        # Assuming 'level' is an attribute of the node indicating its depth or similar
        assert isinstance(node, Node), "input type needs to be of Node"

        if hasattr(node, self.strategy):
            return getattr(node, self.strategy)
        else:
            raise AttributeError(f"Node does not have attribute '{self.strategy}'")

    def _raw(self, node: Node) -> float | int | NDArray:
        return self._leafFunc(node=node)
