import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from .weights import Weights


def getWindows1D(input: NDArray, kernelSize: int, stride: int = 1, padding: int = 0, dilation: int = 0) -> NDArray:
    """
    creates windows of input for convolution and pooling layer
    this function is needed to avoid loops
    """
    batch_size, channels, length = input.shape

    if dilation > 0:
        input = np.insert(input, [i + 1 for i in range(dilation, length - 1, dilation)], 0, axis=2)

    if padding > 0:
        input = np.pad(input, pad_width=((0, 0), (0, 0), (padding, padding)), mode='constant', constant_values=0)

    output_length = (length + 2 * padding - kernelSize) // stride + 1
    batch_strides, channel_strides, length_strides = input.strides
    striding = (batch_strides, channel_strides, stride * length_strides, length_strides)

    return np.lib.stride_tricks.as_strided(input, (batch_size, channels, output_length, kernelSize), striding)


class Convolution1D(Layer):
    """
    a 1D convolution implementation, this can be used for simple time series analysis
    """
    def __init__(self, inChannels: int, outChannels: int, kernelSize: int, stride: int = 1, padding: int= 0, dilation: int = 0, weights: NDArray | None = None, bias: NDArray | None = None) -> None:
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # learnable parameters
        self.weights = Weights((outChannels, inChannels, kernelSize), values=weights)
        self.bias = Weights(outChannels, init='zeros', values=bias)
        self.input = None
        self.windows = None

    def params(self) -> list[Weights]:
        """
        returns weights and bias in a python list, called by optimizers
        """
        return [self.weights, self.bias]

    def forward(self, input: NDArray) -> NDArray:
        """
        The forward pass of convolution
        """
        self.input = input.copy()

        self.windows = getWindows1D(input, self.kernelSize, stride=self.stride, padding=self.padding, dilation=self.dilation)
        output = np.einsum('bclw,ocw->bol', self.windows, self.weights.values) + self.bias.values[None, :, None]

        return output

    def backward(self, gradient: NDArray) -> NDArray:
        """
        The backward pass of convolution
        """
        padding = (self.kernelSize - 1) * (self.dilation + 1)

        gradientWindows = getWindows1D(gradient, self.kernelSize, padding=padding, stride=1, dilation=self.dilation)
        rotatedKernel = np.flip(self.weights.values, axis=2)

        self.weights.deltas = np.einsum('bclw,bol->ocw', self.windows, gradient)
        self.bias.deltas = np.sum(gradient, axis=(0, 2))

        return np.einsum('bclw,ocw->bol', gradientWindows, rotatedKernel)

    def __str__(self) -> str:
        printString = 'Conv1DLayer'
        printString += '    input channels: ' + str(self.inChannels)
        printString += '    output channels: ' + str(self.outChannels)
        printString += '    kernel size: ' + str(self.kernelSize)
        printString += '    padding: ' + str(self.padding)
        printString += '    stride: ' + str(self.stride)
        return printString
