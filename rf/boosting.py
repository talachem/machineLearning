import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from machineLearning.rf.impurityMeasure import Entropy, Gini
from .decisionTree import DecisionTree
from ..data.dataLoader import DataSet


class Boosting(ABC):
    """
    Abstract base class for implementing boosting algorithms.
    """
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @property
    def qualifiedName(self) -> tuple:
        """Returns the fully qualified name of the class."""
        return self.__class__.__module__, self.__class__.__name__

    def toDict(self) -> dict:
        saveDict = {}
        return saveDict

    @classmethod
    def fromDict(cls, loadDict) -> 'Boosting':
        instance = cls()
        return instance

    @abstractmethod
    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the decision tree on the data and targets.
        """
        pass

    @abstractmethod
    def eval(self, trees: list[DecisionTree], data: DataSet) -> NDArray:
        """
        Make predictions for a list of trees, trained by this booster.
        """
        pass


class AdaBoosting(Boosting):
    """
    Adaptive Boosting (AdaBoost) implementation.
    """
    def __init__(self, epsilon: float = 0.01) -> None:
        super().__init__()
        self.alphas = {}
        self.epsilon = epsilon # stability parameter

    def toDict(self) -> dict:
        saveDict = {}
        saveDict['alphas'] = self.alphas
        return saveDict

    @classmethod
    def fromDict(cls, loadDict) -> 'Boosting':
        instance = cls()
        instance.alphas = loadDict['alphas']
        return instance

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the AdaBoost model using a DecisionTree.
        """
        if data.weights is None:
            data.weights = np.ones(len(data)) / len(data)
        tree.train(data)
        self.updateWeights(tree, data)

    def eval(self, trees: list[DecisionTree], data: DataSet) -> NDArray:
        predictions = []
        for tree in trees:
            # Make predictions on the input data
            pred = tree.eval(data)
            # Append predictions to storage list with their respective weights
            alpha = self.alphas[tree.id]
            predictions.append(alpha * pred.reshape(-1, 1))
        return np.sum(predictions, axis=0)

    def updateWeights(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Updates the data weights based on the prediction errors.
        """
        predictions = tree.eval(data)
        errorRate = np.sum(data.weights[data.targets != predictions]) / np.sum(data.weights)
        treeWeights = 0.5 * np.log((1 - errorRate + self.epsilon) / (errorRate + self.epsilon))
        self.alphas[tree.id] = (float(treeWeights))
        data.weights[data.targets == predictions] *= np.exp(-treeWeights)
        data.weights[data.targets != predictions] *= np.exp(treeWeights)
        data.weights /= np.sum(data.weights)


class GradientBoosting(Boosting):
    """
    Gradient Boosting implementation for regression.
    """
    def __init__(self, learningRate: float = 0.1) -> None:
        super().__init__()
        self.residuals = None  # stores the summation of all errors
        self.learningRate = learningRate  # Learning rate for boosting

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the GradientBoosting model using a DecisionTree.
        """
        if self.residuals is None:
            tree.train(data)  # Assuming data includes targets
            self.residuals = data.targets - tree.eval(data)
        else:
            data.targets += self.learningRate * self.residuals
            tree.train(data)  # Train on original data but with new targets
            self.residuals = data.targets - tree.eval(data)

    def eval(self, trees: list[DecisionTree], data: DataSet) -> NDArray:
        predictions = np.zeros(data.shape[0])
        for tree in trees:
            # Make predictions on the input data
            pred = tree.eval(data)
            # Update overall predictions with learning rate
            predictions += self.learningRate * pred
        return predictions


class ProbabilityBoosting(Boosting):
    """
    Gradient Boosting implementation for classification.
    """
    def __init__(self, learningRate: float = 0.1) -> None:
        super().__init__()
        self.residuals = None  # stores the summation of all errors
        self.predictions = {}  # stores past predictions
        self.counter = 0  # counts time steps
        self.learningRate = learningRate  # Learning rate for boosting

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the GradientBoosting model using a DecisionTree.
        """
        if self.residuals is None:
            tree.train(data)
            self.residuals = data.targets - tree.eval(data)
        else:
            probabilities = 1 / (1 + np.exp(-self.predictions[self.counter - 1]))  # Sigmoid for logistic regression
            self.residuals = data.targets - probabilities  # Pseudo-residuals

        self.predictions[self.counter] = tree.eval(data)
        self.counter += 1

    def eval(self, trees: list[DecisionTree], data: DataSet) -> NDArray:
        logits = np.zeros(data.shape[0])
        for tree in trees:
            # Make predictions on the input data
            pred = tree.eval(data)
            # Update overall logits with learning rate
            logits += self.learningRate * pred
        return 1 / (1 + np.exp(-logits))  # Convert logits to probabilities
