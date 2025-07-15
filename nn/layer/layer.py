from abc import ABC, abstractmethod
from numpy.typing import NDArray
from .weights import Weights
import numpy as np


class Layer(ABC):
    """
    this is an abstract class and can only be used indirectly through inherited classes
    """
    __slots__ = ['name', 'mode', 'layerID']
    id = 0

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.mode = ''
        self.layerID = Layer.id
        Layer.id += 1

    @property
    def qualifiedName(self) -> tuple:
        return self.__class__.__module__, self.__class__.__name__

    @abstractmethod
    def forward(self, input: NDArray) -> NDArray:
        """
        it's an abstract method, thus forcing the coder to implement it in daughter classes
        """
        pass

    def __call__(self, *args: NDArray) -> NDArray:
        """
        this is used to make layers behave more like functions
        """
        return self.forward(*args)

    @abstractmethod
    def backward(self, gradient: NDArray) -> NDArray:
        """
        it's an abstract method, thus forcing the coder to implement it in daughter classes
        """
        pass

    def train(self) -> None:
        """
        used to put layer in to training mode
        meaning unfreezes parameters
        """
        self.mode = 'train'

    def eval(self) -> None:
        """
        used to put layer in to evaluation mode
        meaning freezes parameters
        """
        self.mode = 'eval'

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        return self.name

    def quatize(self, bits: int = 8, scheme: str = "symmetric", *agrs, **kwargs) -> None:
        if hasattr(self, "params"):
            for param in self.params():
                param.quantize(bits=bits, scheme=scheme)
