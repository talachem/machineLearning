from numpy.typing import NDArray
import numpy as np
from .layer import Layer
from .weights import Weights


def checkDims(input: NDArray) -> None:
    """
    checks shape/dim for linear layer input
    """
    assert input.ndim == 2, f"Input input should have 2 dimensions, got {input.ndim}"
    batchsize, numFeatures = input.shape
    assert batchsize > 0 and numFeatures > 0, "All dimensions should be greater than 0"


class Linear(Layer):
    """
    linear, dense or mlp layer, multiplies a weight matrix and adds bias
    """
    __slots__ = ['inputSize', 'outputSize', 'input', 'weights', 'bias']

    def __init__(self, inputSize: int, outputSize: int, weights: NDArray | None = None, bias: NDArray | None = None) -> None:
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weights = Weights((inputSize, outputSize), values=weights)
        self.bias = Weights((1, outputSize), values=bias)

    def params(self) -> list[Weights]:
        """
        returns weights and bias in a python list, called by optimizers
        """
        return [self.weights, self.bias]

    def forward(self, input: NDArray) -> NDArray:
        """
        forward pass of the linear layer
        """
        self.input = input
        checkDims(input)
        output = np.matmul(self.input, self.weights.values)
        if self.bias is not False:
            output += self.bias.values
        return output

    def backward(self, gradient: NDArray) -> NDArray:
        """
        backward pass of the linear layer
        """
        self.weights.deltas = np.matmul(self.input.T, gradient)
        if self.bias is not False:
            self.bias.deltas = np.sum(gradient, axis=0, keepdims=True)
        return np.matmul(gradient, self.weights.values.T)

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        printString = self.name
        printString += '    input size: ' + str(self.inputSize)
        printString += '    output size: ' + str(self.outputSize)
        return printString


class Flatten(Layer):
    """
    This layer flattens any given input, the purpose is to use it after a
    convolution block, in order squeeze all channels into one and prepare
    the input for use in a linear layer
    """
    __slots__ = ['inputShape', 'flatShape']
    def __init__(self) -> None:
        super().__init__()
        self.inputShape = None
        self.flatShape = None

    def forward(self, input: NDArray) -> NDArray:
        """
        flattens input into a 1D array, according to batchsize
        """
        if self.inputShape is None:
            self.inputShape = input.shape[1:]
            self.flatShape = np.prod(self.inputShape)
        return input.reshape(-1, self.flatShape)

    def backward(self, gradient: NDArray) -> NDArray:
        """
        unflattens upstream gradient into original input
        """
        return gradient.reshape(-1, *self.inputShape)


class Dropout(Layer):
    """
    dropout layer randomly zeros neurons during forward pass
    and masks the gradient accordingly on the backward pass
    this is used to prevent overfitting
    """
    __slots__ = ['size', 'probability', 'mask']

    def __init__(self, size: int, probability: float) -> None:
        super().__init__()
        self.size = size
        if probability < 0 or probability > 1:
            raise ValueError('probability has to be between 0 and 1')
        self.probability = probability

    def forward(self, input: NDArray) -> NDArray:
        """
        masking input from a linear layer
        """
        checkDims(input)
        if self.mode == 'train':
            self.mask = np.random.random(input.shape) < (1 - self.probability)
            return np.multiply(input, self.mask) / (1 - self.probability)
        else:
            return input

    def backward(self, gradient: NDArray) -> NDArray:
        """
        # masking gradient from a linear layer
        """
        return np.multiply(gradient, self.mask) / (1 - self.probability)

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        printString = self.name
        printString += '    size: ' + str(self.size)
        printString += '    probability: ' + str(self.probability)
        return printString
1
