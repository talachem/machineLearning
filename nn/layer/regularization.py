import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from .layer import Layer
#from .module import Module
from .weights import Weights


class Regularization(Layer):
    """
    regularization layer, this has no effect on the forward pass
    rescales the gradient according to layer weights
    """
    def __init__(self, layers: list, lambda_: float) -> None:
        self.name = self.__class__.__name__
        self.lambda_ = lambda_
        self.params = []

    def forward(self, input: np.ndarray) -> np.ndarray:
        return input

    @abstractmethod
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        pass

    def addParams(self, module) -> None:
        for layer in module:
            try:
               params = layer.params()
               self.params.extend(params)
            except AttributeError:
                continue


class L1Regularization(Regularization):
    """
    L1 implementation of regularization
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, gradient: NDArray) -> NDArray:
        # Compute regularization gradients and add to existing gradients
        for param in self.params:
            gradient += self.lambda_ * np.sign(param.values)
        return gradient


class L2Regularization(Regularization):
    """
    L2 implementation of regularization
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, gradient: NDArray) -> NDArray:
        # Compute regularization gradients and add to existing gradients
        for param in self.params:
            gradient += self.lambda_ * 2 * param.values
        return gradient
