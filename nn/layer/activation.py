import numpy as np
from .layer import Layer
from abc import abstractmethod
from numpy.typing import NDArray


class Activation(Layer):
    """
    the main activation function class containing all the methods used for activation function
    it's an abstract class, meaning it should never be used directly, but instead used a base
    """
    __slots__ = ['input', 'activation', 'useQuantization', 'bits', 'lut', 'inputRange', 'scheme', 'quantOffset']

    def __init__(self) -> None:
        super().__init__()
        self.useQuantization = False
        self.bits = 8
        self.lut = None
        self.inputRange = (0, 1)
        self.scheme = 'asymmetric'
        self.quantOffset = 0

    def forward(self, input: NDArray) -> NDArray:
        """
        Creates the activation and introduces non-linearity to the network.
        Uses a lookup table (LUT) for the quantized path.
        """
        self.input = input
        if self.useQuantization:
            if self.lut is None:
                raise ValueError(f"No LUT generated for {self.name} layer {self.layerID}")

            # Assuming symmetric quantization for activation functions
            quantized_indices = np.clip(np.round(input).astype(np.int32) + self.quantOffset, 0, len(self.lut) - 1)

            # Use the indices to look up the activation values in the LUT
            self.activation = self.lut[quantized_indices]
        else:
            # For the non-quantized path, directly compute the activation
            self.activation = self._function(self.input)

        return self.activation

    def backward(self, gradient: NDArray) -> NDArray:
        """
        creates the upstream gradient from input gradient
        """
        return self._derivative() * gradient

    def quantize(self, bits: int = 8, scheme: str = 'symmetric', overWriteScheme: bool = False) -> None:
        # Initialization steps...
        self.bits = bits
        if overWriteScheme:
            self.scheme = scheme
        minValue, maxValue = self.inputRange

        if self.scheme == "symmetric":
            maxQuant = 2 ** (bits - 1) - 1
            minQuant = - 2 ** (bits - 1)
        elif self.scheme == "asymmetric":
            maxQuant = 2 ** bits - 1
            minQuant = 0
        else:
            raise ValueError(f"{scheme} is not a recognized quantization scheme")

        # Generate values that cover the full input range
        self.quantOffset = minQuant
        scale = (maxValue - minValue) / (maxQuant - minQuant)
        values = np.linspace(minValue, maxValue, 2**bits)
        # Apply the activation function
        activationValues = self._function(values.reshape(1,-1))
        # Quantize the activation outputs
        self.lut = np.round((activationValues - minValue) / scale).astype(np.int32)
        # Ensure LUT indices are within the valid range
        self.lut = np.clip(self.lut, minQuant, maxQuant).flatten()

        self.useQuantization = True

    @abstractmethod
    def _function(self, input: NDArray) -> NDArray:
        """
        it's abstract method, thus must be implemented individually
        """
        pass

    @abstractmethod
    def _derivative(self) -> NDArray:
        """
        it's abstract method, thus must be implemented individually
        """
        pass


class Relu(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.

    ReLU is a commonly used activation function in neural networks, defined as f(x) = max(0, x).
    It is known to perform well in deep learning models due to its ability to produce sparse representations
    and avoid the vanishing gradient problem.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()
        self.inputRange = (0, 6)

    def _function(self, input: NDArray) -> NDArray:
        return np.maximum(0.0, input)

    def _derivative(self) -> NDArray:
        return np.where(self.input > 0, 1, 0)


class Elu(Activation):
    """
    Exponential Linear Unit (ELU) activation function.
    it accepts a scaling parameter
    """
    __slots__ = ['alpha']

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.inputRange = (-1, 6)

    def _function(self, input: NDArray) -> NDArray:
        return np.where(input <= 0., self.alpha * np.exp(input) - 1, input)

    def _derivative(self) -> NDArray:
        return np.where(self.input > 0, 1, self.alpha * np.exp(self.input))


class LeakyRelu(Activation):
    """
    Leaky ReLU activation function.
    one can set the slope on the negative side
    """
    __slots__ = ['epsilon']

    def __init__(self, epislon: float = 1e-1) -> None:
        super().__init__()
        self.epislon = epislon
        self.inputRange = (-6, 6)
        self.scheme = 'symmetric'

    def _function(self, input: NDArray) -> NDArray:
        input[input <= 0.] *= self.epislon
        return input

    def _derivative(self) -> NDArray:
        return np.where(self.input > 1, 1, self.epislon)


class Tanh(Activation):
    """
    The hyperbolic tangent (tanh) activation function.

    This activation function maps input values to the range (-1, 1). It is commonly used in neural networks due to its
    ability to introduce non-linearity while still being differentiable.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()
        self.inputRange = (-1, 1)
        self.scheme = 'symmetric'

    def _function(self, input: NDArray) -> NDArray:
        return np.tanh(input)

    def _derivative(self) -> NDArray:
        return 1 - np.square(self.activation)


class Sigmoid(Activation):
    """
    Sigmoid activation function class.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()

    def _function(self, input: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-input))

    def _derivative(self) -> NDArray:
        return (1 - self.activation) * self.activation


class SoftMax(Activation):
    """
    Softmax activation function.

    Softmax function normalizes the output of a neural network to a probability
    distribution over the classes in the output layer. It is commonly used in
    multi-class classification tasks.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()

    def _function(self, input: NDArray) -> NDArray:
        input = input - np.max(input)
        output = np.exp(input)
        return output/np.sum(output, axis=1, keepdims=True)

    def _derivative(self) -> NDArray:
        return self.activation * (1 - self.activation)


class SoftPlus(Activation):
    """
    The SoftPlus activation function is defined as log(1 + e^x).
    This function is used to introduce non-linearity to a neural network's output.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()
        self.inputRange = (0, 6)

    def _function(self, input: NDArray) -> NDArray:
        return np.log(1. + np.exp(input))

    def _derivative(self) -> NDArray:
        output = np.exp(self.input)
        return output / (1. + output)


class SoftSign(Activation):
    """
    SoftSign activation function.

    The SoftSign activation function maps the input to the range [-1, 1],
    making it useful in neural networks where it is important to limit the range
    of activations to avoid saturation.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()
        self.inputRange = (-1, 1)
        self.scheme = 'symmetric'

    def _function(self, input: NDArray) -> NDArray:
        return input / (np.abs(input) + 1.)

    def _derivative(self) -> NDArray:
        output = np.abs(self.input) + 1.
        return 1. / (output ** 2)


class Identity(Activation):
    """
    The identity activation function.

    The identity function simply returns its input without any transformation.
    It is often used as the activation function for the output layer of a neural network
    when the task involves regression, i.e., predicting a continuous output value.
    """
    __slots__ = []

    def __init__(self) -> None:
        super().__init__()
        self.inputRange = (-6, 6)
        self.scheme = 'symmetric'

    def _function(self, input: NDArray) -> NDArray:
        return input

    def _derivative(self) -> NDArray:
        return 1
