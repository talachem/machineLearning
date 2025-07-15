from abc import ABC, abstractmethod
import numpy as np
from typing import Iterator
from importlib import import_module
from inspect import signature
from numpy.typing import NDArray
from ..layer import Layer, Regularization


class Module(ABC):
    """
    a class for organizing layers of a neural network
    think of a fancy list, that can handle backward and forward passes
    """
    __slots__ = ['name', 'layers', '_index', 'maxIndex']

    def __init__(self, layers: list | None = None) -> None:
        self.name = self.__class__.__name__
        self.layers = [] if layers is None else layers
        self._index = 0
        self.maxIndex = 0 if layers is None else len(layers)

    @property
    def qualifiedName(self) -> tuple:
        return self.__class__.__module__, self.__class__.__name__

    def toDict(self) -> dict:
        saveDict = {}
        for id, layer in enumerate(self):
            saveDict[id] = {}
            saveDict[id]['qualifiedName'] = layer.qualifiedName
            try:
                saveDict[id]['weights'] = layer.weights.toDict()
                saveDict[id]['bias'] = layer.bias.toDict()
            except AttributeError:
                pass
            params = list(signature(type(layer)).parameters)
            saveDict[id]['arguments'] = {}
            for param in params:
                attr = getattr(layer, param)
                if isinstance(attr, (str, int, float, list, tuple)):
                    saveDict[id]['arguments'][param] = attr
                elif isinstance(attr, np.ndarray):
                    saveDict[id]['arguments'][param] = list(attr)
        return saveDict

    @classmethod
    def fromDict(cls, loadDict: dict) -> object:
        instance = cls()
        for id in loadDict:
            moduleName, layerName = loadDict[id]['qualifiedName']
            Module = import_module(moduleName)
            Class = getattr(Module, layerName)
            newLayer = Class(**loadDict[id]['arguments'])
            if 'weights' in loadDict[id]:
                newLayer.weights.fromDict(loadDict[id]['weights'])
                newLayer.bias.fromDict(loadDict[id]['bias'])
            instance.append(newLayer)
        return instance

    def append(self, layer: Layer) -> None:
        """
        appends layer to the layers list
        """
        if isinstance(layer, Regularization):
            layer.addParams(self)
        if isinstance(layer, (Layer, Module)):
            self.layers.append(layer)
        else:
            raise TypeError("is not of type Layer or Module")

    def insert(self, index: int, layer: Layer) -> None:
        """
        insert layer at index to the layers list
        """
        if isinstance(layer, Regularization):
            layer.addParams(self)
        if isinstance(layer, (Layer, Module)):
            self.layers.insert(index, layer)
        else:
            raise TypeError("is not of type Layer or Module")

    def pop(self, index: int) -> None:
        self.layers.pop(index)

    def train(self) -> None:
        """
        sets every layer in the module into train mode
        """
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        """
        sets every layer in the module into eval mode
        """
        for layer in self.layers:
            layer.eval()

    @abstractmethod
    def forward(self, input: NDArray) -> NDArray:
        """
        running of the layers in the module
        """
        pass

    def __call__(self, input: NDArray) -> NDArray:
        """
        this makes using this class more convenient
        """
        return self.forward(input)

    @abstractmethod
    def backward(self, gradient: NDArray) -> NDArray:
        """
        handeling the backward pass
        """
        pass

    def __len__(self) -> int:
        """
        returns the length/number of layers
        """
        return len(self.layers)

    def __repr__(self) -> str:
        """
        used for printing the layers in a human readable manner
        """
        if len(self.layers) == 0:
            return 'no layers appended yet'
        printString = ''
        for i, layer in enumerate(self.layers):
            printString += f'({i}) ' + str(layer) + '\n'
        return printString

    def __getitem__(self, index: int) -> Layer:
        """
        used for indexing and retrieving layers
        this is mainly here to make iterating
        over it easier
        """
        return self.layers[index]

    def __iter__(self) -> Iterator:
        """
        setting up the iterator
        """
        return iter(self.layers)

    def __reversed__(self) -> Iterator:
        """
        setting up the reversed iterator
        """
        return reversed(self.layers)
