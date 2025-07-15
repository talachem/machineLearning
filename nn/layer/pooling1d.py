import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from abc import ABC, abstractmethod
from .convolution1D import getWindows1D
from .linear import checkDims


class Pooling1D(Layer, ABC):
    """
    Abstract base class for 1D pooling layers
    """
    def __init__(self, kernelSize: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernelSize = kernelSize
        self.stride = stride
        self.size = None
        self.channels = None
        self.out = None

    @abstractmethod
    def _function(self) -> NDArray:
        raise NotImplementedError('this needs to be implemented')

    @abstractmethod
    def _derivative(self) -> NDArray:
        raise NotImplementedError('this needs to be implemented')

    def forward(self, input: NDArray) -> NDArray:
        """
        The forward pass of 1D pooling
        """
        self.input = input
        checkDims(input)
        self.batchSize = input.shape[0]

        # setting output sizes
        if self.out is None:
            _, self.channels, self.size = input.shape
            self.out = int((self.size - self.kernelSize) / self.stride) + 1

        # setting output size for backward pass
        self.outputShape = (self.batchSize, self.channels, self.out)

        self.output = getWindows1D(input, self.kernelSize, stride=self.stride)
        self.output = self.output.reshape(self.batchSize, -1, self.kernelSize)
        return self._function().reshape(self.batchSize, -1, self.out)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        The backward pass of 1D pooling
        """
        gradient = gradient.reshape(self.batchSize, -1, 1)
        return gradient * self._derivative()


class MaxPooling1D(Pooling1D):
    """
    1D max pooling implementation
    """
    def __init__(self, kernelSize: int = 2, stride: int = 2) -> None:
        super().__init__(kernelSize, stride)

    def _function(self, keepdims: bool = False) -> NDArray:
        return np.max(self.out, axis=2, keepdims=keepdims)

    def _derivative(self) -> NDArray:
        return (self.output == self._function(keepdims=True)).astype(int)


class MinPooling1D(Pooling1D):
    """
    1D min pooling implementation
    """
    def __init__(self, kernelSize: int = 2, stride: int = 2) -> None:
        super().__init__(kernelSize, stride)

    def _function(self, keepdims: bool = False) -> NDArray:
        return np.min(self.out, axis=2, keepdims=keepdims)

    def _derivative(self) -> NDArray:
        return (self.output == self._function(keepdims=True)).astype(int)


class AvgPooling1D(Pooling1D):
    """
    1D mean/avg pooling implementation
    """
    def __init__(self, kernelSize: int = 2, stride: int = 2) -> None:
        super().__init__(kernelSize, stride)

    def _function(self, keepdims: bool = False) -> NDArray:
        return np.mean(self.out, axis=2, keepdims=keepdims)

    def _derivative(self) -> NDArray:
        return np.ones_like(self.output) / self.kernelSize
