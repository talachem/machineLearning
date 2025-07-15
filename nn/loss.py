import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class LossFunction(ABC):
    """
    Base class for all loss functions.
    """
    __slots__ = ['name', 'prediction', 'target']

    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def forward(self, prediction: NDArray, target: NDArray) -> float:
        """
        Computes the loss for the given prediction and target values.
        """
        self.prediction = prediction
        self.target = target
        loss = self._function(prediction, target)
        return float(np.mean(loss))

    def __call__(self, prediction: NDArray, target: NDArray) -> float:
        """
        Allows the instance of the LossFunction class to be called like a function.
        """
        return self.forward(prediction, target)

    def backward(self) -> NDArray:
        """
        Computes the derivative of the loss with respect to the predicted values.
        """
        return self._derivative()

    @abstractmethod
    def _function(self, prediction: NDArray, target: NDArray) -> NDArray:
        """
        Computes the loss for the given prediction and target values.
        """
        pass

    @abstractmethod
    def _derivative(self) -> NDArray:
        """
        Computes the derivative of the loss with respect to the predicted values.
        """
        pass

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        return self.name


class MAELoss(LossFunction):
    """
    Mean Absolute Error (MAE) loss function for regression tasks.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()

    def _function(self, prediction: NDArray, target: NDArray) -> NDArray:
        """
        Computes the mean absolute error between the predicted and target values.
        """
        return np.abs(target - prediction)

    def _derivative(self) -> NDArray:
        """
        Computes the derivative of the mean absolute error with respect to the predicted values.
        """
        return np.sign(self.prediction - self.target) / self.target.size


class MSELoss(LossFunction):
    """
    Mean Squared Error (MSE) loss function for regression tasks.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()

    def _function(self, prediction: NDArray, target: NDArray) -> NDArray:
        """
        Computes the mean squared error between the predicted and target values.
        """
        return np.power(target - prediction, 2)

    def _derivative(self) -> NDArray:
        """
        Computes the derivative of the mean squared error with respect to the predicted values.
        """
        return 2 * (self.prediction - self.target) / self.target.size


class HuberLoss(LossFunction):
    """
    Class for implementing the Huber loss function for regression tasks.
    """
    __slots__ = ['delta']

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def _function(self, prediction: NDArray, target: NDArray) -> NDArray:
        """
        Calculates the Huber loss value based on the prediction and target arrays.
        """
        diff = target - prediction
        return np.where(np.abs(diff) < self.delta, 0.5 * np.square(diff), self.delta * np.abs(diff) - 0.5 * self.delta ** 2)

    def _derivative(self) -> NDArray:
        """
        Calculates the derivative of the Huber loss function with respect to the prediction array.
        """
        diff = self.prediction - self.target
        return np.where(np.abs(diff) < self.delta, diff, self.delta * np.sign(diff))


class NLLLoss(LossFunction):
    """
    intended for classification
    """
    __slots__ = ['epsilon']

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def _function(self, prediction: NDArray, target: NDArray) -> NDArray:
        """
        Compute the negative log-likelihood loss for binary classification.
        """
        return - (target * np.log(prediction + self.epsilon) + (1 - target) * np.log(1 - prediction + self.epsilon))

    def _derivative(self) -> NDArray:
        """
        Compute the derivative of the negative log-likelihood loss.
        """
        return (self.prediction - self.target) / (self.prediction * (1 - self.prediction) + self.epsilon)


class CrossEntropyLoss(LossFunction):
    """
    intended for classification
    """
    __slots__ = ['epsilon']

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon # stability parameter

    def _function(self, prediction: NDArray, target: NDArray) -> NDArray:
        """
        Compute cross entropy loss for binary classification.
        """
        confidence = np.sum(prediction * target, axis=1)
        return - np.log(confidence + self.epsilon)

    def _derivative(self) -> NDArray:
        """
        Compute the derivative of cross entropy loss.
        """
        return (self.prediction - self.target) / self.target.size


class FocalLoss(LossFunction):
    """
    intended for classification
    this is an alternative to cross entropy loss...
    it could be that the derivative is wrong
    """
    __slots__ = ['focus', 'epsilon']

    def __init__(self, focus: float = 1.5, epsilon: float = 1e-8):
        super().__init__()
        self.focus = focus
        self.epsilon = epsilon # stability parameter

    def _function(self, prediction: NDArray, target: NDArray) -> NDArray:
        """
        Compute focal loss for binary classification.
        """
        confidence = np.sum(prediction * target, axis=1)
        return - self._power(1 - confidence, self.focus) * np.log(confidence + self.epsilon)

    def _derivative(self) -> NDArray:
        """
        Compute the derivative of focal loss.
        """
        pt = np.sum(self.prediction * self.target, axis=1) # p_t
        term1 = - (1 - pt)**self.focus * (-np.log(pt) - 1)
        term2 = self.focus * (1 - pt)**(self.focus - 1) * np.log(pt)
        return - (term1 + term2)


class HellingerLoss(LossFunction):
    """
    intended for classification
    """
    __slots__ = ['epsilon']

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon # stability parameter

    def _function(self, prediction: NDArray, target: NDArray) -> NDArray:
        """
        Compute hellinger loss for binary classification.
        """
        return np.power(np.sqrt(target) - np.sqrt(prediction), 2)

    def _derivative(self) -> NDArray:
        """
        Compute the derivative of hellinger loss.
        """
        p_sqrt = np.sqrt(self.prediction)
        q_sqrt = np.sqrt(self.target)
        return (1/2*np.sqrt(2)) * (q_sqrt - p_sqrt) / p_sqrt
