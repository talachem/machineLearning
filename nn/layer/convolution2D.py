import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from .weights import Weights


def assignParameter(parameter: int | tuple) -> tuple:
    """
    this checks wether a parameter is a tuple or int and returns a tuple
    """
    if isinstance(parameter, (int, float)):
        if float(parameter).is_integer():
            return (int(parameter), int(parameter))
        else:
            raise ValueError('the parameter should be a whole number')
    return parameter


def getWindows(input: NDArray, kernelSize: tuple[int, int], outputSize: tuple[int, int, int, int], padding: tuple[int, int] = (0,0), stride: tuple[int, int] = (1,1), dilate: tuple[int, int] = (0,0)) -> NDArray:
    """
    creates windows of input for convolution and pooling layer
    this function is needed to avoid loops
    """

    # getting shape parameters
    batchSize, channels, height, width = input.shape

    # dilate the input if necessary
    if dilate[0] != 0:
        input = np.insert(input, range(1, height), 0, axis=2)
    if dilate[1] != 0:
        input = np.insert(input, range(1, width), 0, axis=3)

    # pad the input if necessary
    if padding[0] != 0 or padding[1] != 0:
        input = np.pad(input, pad_width=((0,), (0,), (padding[0],), (padding[1],)), mode='constant', constant_values=(0.,))

    # getting the strides of input
    batchStrides, channelStrides, kernelHeightStrides, kernelWidthStrides = input.strides
    striding = (batchStrides, channelStrides, stride[0] * kernelHeightStrides, stride[1] * kernelWidthStrides, kernelHeightStrides, kernelWidthStrides)

    # returning the windows
    return np.lib.stride_tricks.as_strided(input, (*outputSize, kernelSize[0], kernelSize[1]), striding)


def checkDims(input: NDArray) -> None:
    """
    Checks that the input tensor has the correct shape.
    """

    # Check that the input tensor has 4 dimensions.
    assert input.ndim == 4, f"Input tensor should have 4 dimensions, got {input.ndim}"

    # Get the size of each dimension.
    batchsize, channels, height, width = input.shape

    # Check that all dimensions have a size greater than 0.
    assert batchsize > 0 and channels > 0 and height > 0 and width > 0, "All dimensions should be greater than 0"


class Convolution2D(Layer):
    """
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    """
    __slots__ = ['inChannels', 'outChannels', 'kernelSize', 'stride', 'padding', 'windows', 'input', 'weights', 'bias']

    def __init__(self, inChannels: int, outChannels: int, kernelSize: tuple = (3,3), padding: tuple = (0,0), stride: tuple = (1,1), weights: NDArray | None = None, bias: NDArray | None = None) -> None:
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels

        self.kernelSize = assignParameter(kernelSize)
        self.stride = assignParameter(stride)
        self.padding = assignParameter(padding)

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
        The forward pass of convolution
        """
        self.input = input
        checkDims(input)

        batchSize, channels, height, width = input.shape
        outHeight = (height - self.kernelSize[0] + 2 * self.padding[0]) / self.stride[0] + 1
        outWidth = (width - self.kernelSize[1] + 2 * self.padding[1]) / self.stride[1] + 1
        outputSize = (batchSize, channels, int(outHeight), int(outWidth))

        self.windows = getWindows(input, self.kernelSize, outputSize, self.padding, self.stride)
        output = np.einsum('bihwkl,oikl->bohw', self.windows, self.weights.values) + self.bias.values[None, :, None, None]
        return output

    def backward(self, gradient: NDArray) -> NDArray:
        """
        The backward pass of convolution
        """
        padding = (self.kernelSize[0] - 1, self.kernelSize[1] - 1) if self.padding == 0 else self.padding

        gradientWindows = getWindows(gradient, self.kernelSize, self.input.shape, padding=padding, stride=(1,1), dilate=(self.stride[0] - 1, self.stride[1] - 1))
        rotatedKernel = np.rot90(self.weights.values, 2, axes=(2, 3))

        self.weights.deltas = np.einsum('bihwkl,bohw->oikl', self.windows, gradient)
        self.bias.deltas = np.sum(gradient, axis=(0, 2, 3))

        return np.einsum('bohwkl,oikl->bihw', gradientWindows, rotatedKernel)

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        printString = self.name
        printString += '    input channels: ' + str(self.inChannels)
        printString += '    output channels: ' + str(self.outChannels)
        printString += '    kernel size: ' + str(self.kernelSize)
        printString += '    padding: ' + str(self.padding)
        printString += '    stride: ' + str(self.stride)
        return printString


class Unsqueeze(Layer):
    """
    this layer type exists because I was too lazy adding/removing .reshape
    to inputs, depending if there is a convolution or not as the first layer
    it reshapes the input according to user specification
    if no channel information is given, the class assumes 1 channel
    """
    __slots__ = ['inputShape', 'orginialShape']

    def __init__(self, inputShape: tuple[int, int, int]) -> None:
        super().__init__()

        # testing if inputShape provides (channels, height, width)
        if isinstance(inputShape, list):
            inputShape = tuple(inputShape)
        if not isinstance(inputShape, tuple):
            raise TypeError('input shape should be a tuple (or list)')
        if len(inputShape) == 2:
            inputShape = (1, *inputShape)
        elif len(inputShape) < 2 or len(inputShape) > 3:
            raise ValueError('input shape not corrisponding to (channels, height, width)')

        # class attributes
        self.inputShape = inputShape
        self.orginialShape = None

    def forward(self, input: NDArray) -> NDArray:
        """
        Reshapes input into an acceptable shape for convolutions
        """
        if self.orginialShape is None:
            self.orginialShape = input.shape[1:]
        return input.reshape(-1, *self.inputShape)

    def backward(self, gradient: NDArray) -> NDArray:
        """
        Reshapes the upstream gradient into original shape
        """
        return gradient.reshape(-1, *self.orginialShape)

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        printString = self.name
        printString += '    into shape: ' + str(self.inputShape)
        return printString
