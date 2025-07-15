import numpy as np
from numpy._typing import NDArray
from .layer import Layer
from .weights import Weights


class BatchNorm2D(Layer):
    """
    normalizes 2d input of shape (batchsize, channels, height, width)
    """
    __slots__ = ['input', 'channels', 'runningMean', 'runningVariance', 'momentum', 'epsilon', 'batchSize', 'mean', 'variance', 'tiledMean', 'tiledVariance', 'normalized', 'weights', 'bias', 'inputShape']

    def __init__(self, inputShape: tuple[int, int, int], momentum: float = 0.1, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.channels = inputShape[0]
        self.weights = Weights(inputShape, init='ones')
        self.bias = Weights(inputShape[0], init='zeros')
        self.runningMean = np.zeros(inputShape[0])
        self.runningVariance = np.zeros(inputShape[0])
        self.momentum = momentum
        self.epsilon = epsilon
        self.inputShape = inputShape

    def params(self) -> list[Weights]:
        """
        returns weights and bias in a python list, called by optimizers
        """
        return [self.weights, self.bias]

    def forward(self, input: NDArray) -> NDArray:
        """
        normalizes the input with running averges
        this is a learning layer, thus it also has weights and biases
        it maintains image size
        """
        self.input = input
        self.batchSize = input.shape[0]

        self.mean = self.input.mean(axis=(0,2,3))
        self.variance = np.sqrt((self.input.var(axis=(0,2,3))) + self.epsilon) # epsilon is used to avoid divisions by zero
        self._runningVariables()

        self.tiledMean = np.tile(self.mean, self.batchSize).reshape(self.batchSize, self.channels, 1, 1)
        self.tiledVariance = np.tile(self.variance, self.batchSize).reshape(self.batchSize, self.channels, 1, 1)

        self.normalized = (self.input - np.tile(self.runningMean,self.batchSize).reshape(self.batchSize, self.channels, 1, 1)) / np.tile(self.runningVariance,self.batchSize).reshape(self.batchSize, self.channels, 1, 1)
        return self.weights.values * self.normalized + self.bias.values[None,:,None,None]

    def _runningVariables(self) -> None:
        """
        this is called during 'forward' method
        it mixes the running avergaes with current averages
        """
        if self.mode == 'train':
            self.runningMean = self.momentum * self.runningMean + (1 - self.momentum) * self.mean
            self.runningVariance = self.momentum * self.runningVariance + (1 - self.momentum) * self.variance

        # the next two if statements are there in case someone calls the network in eval mode without prior training
        if self.mode == 'eval' and np.sum(self.runningMean) == 0:
            self.runningMean = self.mean
        if self.mode == 'eval' and np.sum(self.runningVariance) == 0:
            self.runningVariance = self.variance

    def backward(self, gradient: NDArray) -> NDArray:
        """
        transforms upstream gradient, the effect of variance and mean
        calculates deltas for weights and biases
        its input and output size are the same
        """
        self.weights.deltas = (gradient * self.normalized).sum(axis=0)
        self.bias.deltas = gradient.sum(axis=(0,2,3))

        gradient = gradient * self.weights.values

        deltaMean = (gradient / (-self.tiledVariance)).mean(0)
        deltaVariance = ((gradient * (self.input - self.tiledMean)).sum(0) * ((-.5 / self.tiledVariance) ** 3))

        gradient = (gradient / self.tiledVariance + deltaVariance * 2 * (self.input - self.tiledMean) / self.mean.size + deltaMean / self.mean.size)
        return gradient


class BatchNorm1D(Layer):
    """
    normalizes 1d input of shape (batchsize, whatever)
    """
    __slots__ = ['input', 'runningMean', 'runningVariance', 'momentum', 'epsilon', 'batchSize', 'mean', 'variance', 'tiledMean', 'tiledVariance', 'normalized', 'weights', 'bias', 'numFeatures']

    def __init__(self, numFeatures: int, momentum: float = 0.1, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.weights = Weights((1, numFeatures), init='ones')
        self.bias = Weights((1, numFeatures), init='zeros')
        self.runningMean = 0
        self.runningVariance = 0
        self.momentum = momentum
        self.epsilon = epsilon
        self.numFeatures = numFeatures

    def params(self) -> list[Weights]:
        """
        returns weights and bias in a python list, called by optimizers
        """
        return [self.weights, self.bias]

    def forward(self, input: NDArray) -> NDArray:
        """
        normalizes the input with running averges
        this is a learning layer, thus it also has weights and biases
        it maintains input size
        """
        self.input = input
        self.batchSize = input.shape[0]

        self.mean = self.input.mean(axis=0)
        self.variance = np.sqrt((self.input.var(axis=(0))) + self.epsilon) # epsilon is used to avoid divisions by zero
        self._runningVariables()

        self.normalized = (self.input - self.runningMean) / self.runningVariance
        return self.weights.values * self.normalized + self.bias.values

    def _runningVariables(self) -> None:
        """
        this is called during 'forward' method
        it mixes the running avergaes with current averages
        """
        if self.mode == 'train':
            self.runningMean = self.momentum * self.runningMean + (1 - self.momentum) * self.mean
            self.runningVariance = self.momentum * self.runningVariance + (1 - self.momentum) * self.variance

        # the next two if statements are there in case someone calls the network in eval mode without prior training
        if self.mode == 'eval' and np.sum(self.runningMean) == 0:
            self.runningMean = self.mean
        if self.mode == 'eval' and np.sum(self.runningVariance) == 0:
            self.runningVariance = self.variance

    def backward(self, gradient: NDArray) -> NDArray:
        """
        transformes upstream gradient, the effect of variance and mean
        calculates deltas for weights and biases
        its input and output size are the same
        """
        self.weights.deltas = np.sum(gradient * self.normalized, axis=0, keepdims=True)
        self.bias.deltas = gradient.sum(axis=0, keepdims=True)

        gradient = gradient * self.weights.values

        deltaVariance = np.sum((gradient * (self.input - self.mean)) * ((-.5 / self.variance) ** 3), axis=0, keepdims=True)
        deltaMean = (gradient / (-self.mean + self.epsilon)).mean(axis=0, keepdims=True)

        gradient = (gradient / self.variance + deltaVariance * 2 * (self.input - self.mean) / self.mean.size + deltaMean / self.input.size)
        return gradient
