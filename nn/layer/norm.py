import numpy as np
from numpy.typing import NDArray
from .layer import Layer


class L1Norm(Layer):
    """
    A basic norming layer, doesn't learn anything or affect
    the learning process beyond norming inputs
    """
    def __init__(self, axis=None, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.axis = axis # Axis or axes along which to compute the L1 norm
        self.scales = None
        self.epsilon = epsilon # A small value to avoid division by zero

    def forward(self, input: NDArray) -> NDArray:
        # Compute L1 norm (sum of absolute values) along the specified axis
        norm = np.abs(input).sum(axis=self.axis, keepdims=True)
        # Compute reciprocal of L1 norm
        norm = 1. / (norm + self.epsilon)
        # Normalize the input
        output = input * norm
        # Compute scale factors (sign of the output)
        self.scales = -np.sign(output)
        # Initialize gradient to zero array of same shape as output
        self.gradient = np.zeros_like(output, dtype=float)
        return output

    def backward(self, gradient: NDArray) -> NDArray:
        # Add scales to the gradient
        self.gradient += self.scales
        # Add gradient to the input gradient
        gradient[:] += self.gradient
        return gradient


class L2Norm(Layer):
    """
    A basic norming layer, doesn't learn anything or affect
    the learning process beyond norming inputs
    """
    def __init__(self, axis=None, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.axis = axis # Axis or axes along which to compute the L2 norm
        self.scales = None
        self.epsilon = epsilon # A small value to avoid division by zero

    def forward(self, input: NDArray) -> NDArray:
        # Compute L2 norm (square root of sum of squares) along the specified axis
        norm = (input * input).sum(axis=self.axis, keepdims=True)
        # Compute reciprocal of L2 norm
        norm = 1. / np.sqrt(norm + self.epsilon)
        # Normalize the input
        output = input * norm
        # Compute scale factors
        self.scales = (1. - output) * norm
        # Initialize gradient to zero array of same shape as output
        self.gradient = np.zeros_like(output, dtype=float)
        return output

    def backward(self, gradient: NDArray) -> NDArray:
        # Add scales to the gradient
        self.gradient += self.scales
        # Add gradient to the input gradient
        gradient[:] += self.gradient
        return gradient
