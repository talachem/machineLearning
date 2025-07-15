import numpy as np
from numpy.typing import NDArray
from .module import Module


class Sequential(Module):
    """
    a list for layers to be called in sequence
    incorporates the forward and backward pass
    """
    __slots__ = []

    def __init__(self, layers: list | None = None) -> None:
        super().__init__(layers)

    def forward(self, input: NDArray) -> NDArray:
        """
        calls all layers sequentially
        """
        for layer in self:
            input = layer(input)
        return input

    def backward(self, gradient: NDArray) -> NDArray:
        """
        calls all layers sequentially in reverse
        """
        for layer in reversed(self):
            gradient = layer.backward(gradient)
        return gradient
