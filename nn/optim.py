import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from .module import Module, Sequential
from .layer import Weights
from typing import Protocol


class LearningLayer(Protocol):
    @abstractmethod
    def params(self) -> list[Weights]:
        ...


class Optimizer(ABC):
    """
    this is the base class for all optimizers
    it takes the parameters (weights/biases) and gradients from layers and optimizes them
    """
    __slots__ = ['name', 'learningRate', 'layers', 'scheduler', 'learning']

    def __init__(self, layers: list | Module, learningRate: float) -> None:
        self.name = self.__class__.__name__
        self.learningRate = learningRate

        # here I make the assumption that the user will want to use a sequential execution of the layers in question
        if isinstance(layers, list):
            self.layers = Sequential(layers)
        elif isinstance(layers, Module):
            self.layers = layers
        else:
            raise TypeError('The layers argument must be either a list or an instance of the Module class.')

    @abstractmethod
    def update(self, params: list[Weights]) -> None:
        """
        implemented according to algorithm for every optimizer
        """
        pass

    def step(self, gradient: NDArray) -> None:
        """
        stepping through the layers in reverse order, calls gradient method for each layer
        """
        _ = self.layers.backward(gradient)
        for layer in reversed(self.layers):
            try:
                params = layer.params()
            except AttributeError:
                # 'params' method not found in the layer, skip updating
                continue

            self.update(params)

        self.postStep()

    def postStep(self) -> None:
        """
        an optional method used for optimizers to perform stuff after updating layer weights
        """
        pass

    def __str__(self) -> str:
        printString = self.name
        printString += '       learningRate: ' + str(self.learningRate)
        #printString += '    learning layers: ' + str(len(self.learning))
        return printString


class SGD(Optimizer):
    """
    Stochastic gradient descent
    """
    __slots__ = []

    def __init__(self, layers: list, learningRate: float) -> None:
        super().__init__(layers, learningRate)

    def update(self, params: list[Weights]) -> None:
        for param in params:
            param.values -= self.learningRate * param.deltas


class SGDMomentum(Optimizer):
    """
    Stochastic gradient descent with momentum on mini-batches.
    """
    __slots__ = ['momentum']

    def __init__(self, layers: list, learningRate: float, momentum: float) -> None:
        super().__init__(layers, learningRate)
        self.momentum = momentum

    def update(self, params: list[Weights]) -> None:
        for param in params:
            if param.prevValues is None:
                param.prevValues = np.zeros(param.values.shape)
            delta = self.learningRate * param.deltas - self.momentum * param.prevValues
            param.values -= delta
            param.prevValues = delta


class NesterovMomentum(Optimizer):
    """
    Stochastic Gradient Descent with Nesterov Momentum specialiation
    """
    __slots__ = ['momentum']

    def __init__(self, layers: list, learningRate: float, momentum: float = .9) -> None:
        super().__init__(layers, learningRate)
        self.momentum = momentum

    def update(self, params: list[Weights]) -> None:
        for param in params:
            if param.prevValues is None:
                param.prevValues = np.zeros(param.values.shape)

            momentum_term = self.momentum * param.prevValues
            param.prevValues = momentum_term - self.learningRate * param.deltas
            param.values += momentum_term - self.learningRate * param.deltas


class AdaGrad(Optimizer):
    """
    AdaGrad adaptive optimisation algorithm
    """
    __slots__ = ['epsilon']

    def __init__(self, layers: list, learningRate: float, epsilon: float = 1e-6) -> None:
        super().__init__(layers, learningRate)
        self.epsilon = epsilon

    def update(self, params: list[Weights]) -> None:
        for param in params:
            if param.cache is None:
                param.cache = {'cache': np.zeros(param.values.shape)}
            param.cache['cache'] += param.deltas ** 2
            param.values += -self.learningRate * param.deltas / (np.sqrt(param.cache['cache']) + self.epsilon)


class AdaDelta(Optimizer):
    """
    AdaDelta optimization algorithm
    """
    __slots__ = ['epsilon']

    def __init__(self, layers: list, learningRate: float, rho: float = 0.9, epsilon: float = 1e-6) -> None:
        super().__init__(layers, learningRate)
        self.rho = rho
        self.epsilon = epsilon

    def update(self, params: list[Weights]) -> None:
        for param in params:
            if param.cache is None:
                param.cache = {'cache': np.zeros(param.values.shape), 'delta': np.zeros(param.values.shape)}
            param.cache['cache'] = self.rho * param.cache['cache'] + (1 - self.rho) * param.deltas ** 2
            update = param.deltas * np.sqrt(param.cache['delta'] + self.epsilon) / np.sqrt(param.cache['cache'] + self.epsilon)
            param.values -= self.learningRate * update
            param.cache['delta'] = self.rho * param.cache['delta'] + (1 - self.rho) * update ** 2


class RMSprop(Optimizer):
    """
    RMSprop adaptive optimization algorithm
    """
    __slots__ = ['decayRate', 'epsilon']

    def __init__(self, layers: list, learningRate: float, decayRate: float = 0.9, epsilon: float = 1e-6) -> None:
        super().__init__(layers, learningRate)
        self.decayRate = decayRate
        self.epsilon = epsilon

    def update(self, params: list[Weights]) -> None:
        for param in params:
            if param.cache is None:
                param.cache = {'cache': np.zeros(param.values.shape)}
            param.cache['cache'] = self.decayRate * param.cache['cache'] + (1 - self.decayRate) * param.deltas ** 2
            param.values += - self.learningRate * param.deltas / (np.sqrt(param.cache['cache']) + self.epsilon)


class Adam(Optimizer):
    """
    Adam optimizer, bias correction is implemented.
    """
    __slots__ = ['t', 'beta1', 'beta2', 'epsilon']

    def __init__(self, layers: list, learningRate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 10e-8) -> None:
        super().__init__(layers, learningRate)
        self.t = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, params: list[Weights]) -> None:
        for param in params:
            if param.cache is None:
                param.cache = {'m': np.zeros(param.values.shape), 'v': np.zeros(param.values.shape)}
            param.cache['m'] = self.beta1 * param.cache['m'] + (1 - self.beta1) * param.deltas
            param.cache['v'] = self.beta2 * param.cache['v'] + (1 - self.beta2) * param.deltas ** 2
            mCorrected = param.cache['m'] / (1 - self.beta1 ** self.t)
            vCorrected = param.cache['v'] / (1 - self.beta2 ** self.t)
            param.values += -self.learningRate * mCorrected / (np.sqrt(vCorrected) + self.epsilon)

    def postStep(self) -> None:
        self.t += 1
