import numpy as np
from numpy.typing import NDArray
from .convolution2D import getWindows, checkDims, assignParameter
from .layer import Layer
from abc import ABC, abstractmethod


class Pooling2D(Layer, ABC):
    """
    pooling layer to be used after convolution
    it can do min/max pooling, avg/mean pooling hasn't been implemented, because backwards pass is a bit more complicated
    """
    def __init__(self, kernelSize: tuple = (2,2), stride: tuple = (2,2)) -> None:
        super().__init__()
        self.kernelSize = assignParameter(kernelSize)
        self.xKern, self.yKern = kernelSize[0], kernelSize[1]
        self.stride = assignParameter(stride)
        self.xStride, self.yStride = stride[0], stride[1]
        self.xSize, self.ySize = None, None
        self.channels = None
        self.xOut, self.yOut = None, None

    @abstractmethod
    def _function(self) -> NDArray:
        raise NotImplementedError('this needs to be implemented')

    @abstractmethod
    def _derivative(self) -> NDArray:
        raise NotImplementedError('this needs to be implemented')

    def forward(self, input: NDArray) -> NDArray:
        """
        The forward pass of pooling
        """
        self.input = input
        checkDims(input)
        self.batchSize = input.shape[0]

        # setting output sizes
        if self.xOut is None or self.yOut is None:
            _, self.channels, self.xSize, self.ySize = input.shape
            self.xOut = int((self.xSize - self.xKern) / self.xStride) + 1
            self.yOut = int((self.ySize - self.yKern) / self.yStride) + 1

        # setting output size for backward pass
        self.outputShape = (self.batchSize, self.channels, self.xOut, self.yOut)

        self.output = getWindows(input, self.kernelSize, self.outputShape, stride=self.stride)
        self.output = self.output.reshape(self.batchSize, self.channels, -1, self.xKern * self.yKern)
        return self._function().reshape(self.batchSize, self.channels, self.xOut, self.yOut)

    def backward(self, gradient: NDArray) -> NDArray:
        """
        The backward pass of pooling
        """
        gradient = gradient.reshape(self.batchSize, self.channels, -1, 1)
        return gradient * self._derivative()


class MaxPooling2D(Pooling2D):
    """
    2D max pooling implementation
    """
    def __init__(self, kernelSize: tuple = (2,2), stride: tuple = (2,2)) -> None:
        super().__init__(kernelSize, stride)

    def _function(self, keepdims: bool = False) -> NDArray:
        return np.max(self.output, axis=3, keepdims=keepdims)

    def _derivative(self) -> NDArray:
        return (self.output == self._function(keepdims=True)).astype(int)


class MinPooling2D(Pooling2D):
    """
    2D min pooling implementation
    """
    def __init__(self, kernelSize: tuple = (2,2), stride: tuple = (2,2)) -> None:
        super().__init__(kernelSize, stride)

    def _function(self, keepdims: bool = False) -> NDArray:
        return np.min(self.output, axis=3, keepdims=keepdims)

    def _derivative(self) -> NDArray:
        return (self.output == self._function(keepdims=True)).astype(int)


class AvgPooling2D(Pooling2D):
    """
    2D mean/avg pooling implementation
    """
    def __init__(self, kernelSize: tuple = (2,2), stride: tuple = (2,2)) -> None:
        super().__init__(kernelSize, stride)

    def _function(self, keepdims: bool = False) -> NDArray:
        return np.mean(self.output, axis=3, keepdims=keepdims)

    def _derivative(self) -> NDArray:
        return np.ones_like(self.output) / (self.xKern * self.yKern)
