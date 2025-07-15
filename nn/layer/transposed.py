import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from .convolution2D import getWindows, assignParameter
from .weights import Weights


class TransposedConv(Layer):
    """
    Transposed Convolution Layer for a Convolutional Neural Network (CNN).
    This layer can increase the spatial dimensions (height, width) of the input.
    It is also known as a deconvolution layer, although it does not exactly reverse the convolution operation.
    """
    __slots__ = ['inChannels', 'outChannels', 'kernelSize', 'stride', 'padding', 'windows', 'input', 'paddingTransposed', 'strideTransposed', 'weights', 'bias']

    def __init__(self, inChannels: int, outChannels: int, kernelSize: tuple = (3,3), padding: tuple = (0,0), stride: tuple = (1,1), weights: NDArray | None = None, bias: NDArray | None = None) -> None:
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels

        self.kernelSize = assignParameter(kernelSize)
        self.paddingTransposed = (kernelSize[0] - padding[0] - 1, kernelSize[1] - padding[1] - 1)
        self.padding = assignParameter(padding)
        self.strideTransposed = (stride[0] - 1, stride[1] - 1)
        self.stride = assignParameter(stride)

        # learnable parameters
        self.weights = Weights((self.outChannels, self.inChannels, self.kernelSize[0], self.kernelSize[1]), values=weights)
        self.bias = Weights(self.outChannels, init='zeros', values=bias)

    def params(self) -> list[Weights]:
        """
        returns weights and bias in a python list, called by optimizers
        """
        return [self.weights, self.bias]

    def forward(self, input: NDArray) -> NDArray:
        """
        The forward pass of transposed convolution
        """
        self.input = input
        batchSize, channels, height, width = input.shape
        outHeight = self.stride[0] * (height - 1) + self.kernelSize[0] - 2 * self.padding[0]
        outWidth = self.stride[1] * (width - 1) + self.kernelSize[1] - 2 * self.padding[1]
        self.outputShape = (batchSize, channels, outHeight, outWidth)
        self.windows = getWindows(self.input, self.kernelSize, self.outputShape, padding=self.paddingTransposed, stride=1, dilate=self.strideTransposed)
        out = np.einsum('bihwkl,oikl->bohw', self.windows, self.weights.values)

        # add bias to kernels
        out += self.bias.values[None, :, None, None]
        return out

    def backward(self, gradient: NDArray) -> NDArray:
        """
        The backward pass of transposed convolution
        """
        gradientWindows = getWindows(gradient, self.kernelSize, self.input.shape, padding=0, stride=self.stride)
        rotatedKernel = np.rot90(self.weights.values, 2, axes=(2, 3))

        self.weights.deltas = np.einsum('bihwkl,bohw->oikl', self.windows, gradient)
        self.bias.deltas = np.sum(gradient, axis=(0, 2, 3))

        return np.einsum('bohwkl,oikl->bihw', gradientWindows, rotatedKernel)
