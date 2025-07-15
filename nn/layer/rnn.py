import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from .weights import Weights
from .linear import Linear
from abc import abstractmethod


def checkDims(input: np.ndarray) -> None:
    """
    checks input shape/dims for RNN layer
    """
    assert input.ndim == 3, f"Input input should have 3 dimensions, got {input.ndim}"
    batchsize, seqLength, _ = input.shape
    assert batchsize > 0 and seqLength > 0, "All dimensions should be greater than 0"


class RNN(Layer):
    """
    abstract RNN layer implementation, used in RNN and LSTM
    """
    __slots__ = ['inputSize', 'hiddenSize', 'outputSize']

    def __init__(self, inputSize: int, hiddenSize: int, outputSize: int) -> None:
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

    @abstractmethod
    def forward(self, input: NDArray, hiddenState: NDArray | None = None) -> tuple[NDArray, NDArray]:
        raise NotImplementedError('not implemented')

    @abstractmethod
    def backward(self, gradient: NDArray, hiddenGradient: NDArray | None = None) -> tuple[NDArray, NDArray]:
        raise NotImplementedError('not implemented')


class Recuring(RNN):
    """
    a concrete implementation of RNN layer
    """
    __slots__ = ['gradientClipSize', 'input', 'batchSize', 'seqLength', 'inputLayer', 'hiddenLayer', 'outputLayer']

    def __init__(self, inputSize: int, hiddenSize: int, outputSize: int, gradientClipSize: int = 5) -> None:
        super().__init__(inputSize, hiddenSize, outputSize)
        self.gradientClipSize = gradientClipSize

        self.inputLayer = Linear(inputSize, hiddenSize, bias=False)
        self.hiddenLayer = Linear(hiddenSize, hiddenSize)
        self.outputLayer = Linear(hiddenSize, outputSize)

    def params(self) -> list[Weights]:
        """
        returns weights and bias in a python list, called by optimizers
        """
        return [*self.inputLayer.params(), *self.hiddenLayer.params(), *self.outputLayer.params()]

    def forward(self, input: NDArray, hiddenState: NDArray | None = None) -> tuple[NDArray, NDArray]:
        """
        forward pass
        """
        checkDims(input)

        self.input = input
        self.batchSize, self.seqLength, _ = input.shape
        outputState = np.zeros((self.batchSize, self.seqLength, self.outputSize))

        # Initialize the hidden state
        if hiddenState is None:
            hiddenState = np.zeros((self.batchSize, self.hiddenSize))
        else:
            # Ensure that the hiddenState has the correct shape
            assert hiddenState.shape == (self.batchSize, self.hiddenSize)

        # Accumulate hidden states over time steps
        hiddenStates = np.zeros((self.batchSize, self.seqLength, self.hiddenSize))

        # Loop through each time step
        for t in range(self.seqLength):
            inputTimeStep = self.inputLayer(input[:, t, :])
            hiddenTimeStep = self.hiddenLayer(hiddenState)
            hiddenState = np.tanh(inputTimeStep + hiddenTimeStep)
            outputState[:, t, :] = self.outputLayer(hiddenState)
            hiddenStates[:, t, :] = hiddenState

        return outputState, hiddenStates

    def backward(self, gradient: NDArray, hiddenGradient: NDArray | None = None) -> tuple[NDArray, NDArray]:
        """
        backwar pass
        """
        gradInputState = np.zeros_like(self.input)
        dhiddenNext = np.zeros((self.batchSize, self.hiddenSize))

        # If hiddenGradient is provided, it should be added to the last timestep's hidden state gradient
        if hiddenGradient is not None:
            dhiddenNext += hiddenGradient

        for t in reversed(range(self.seqLength)):
            gradientTimeStep = gradient[:, t, :]
            dhidden = self.outputLayer.backward(gradientTimeStep) + dhiddenNext
            dhiddenRaw = (1 - np.square(self.hiddenState[:, t, :])) * dhidden
            dhiddenNext = self.hiddenLayer.backward(dhiddenRaw)

            dinput = self.inputLayer.backward(dhiddenRaw)
            gradInputState[:, t, :] = dinput

        # Clip gradients
        for dparam in [self.inputLayer.weights.deltas, self.hiddenLayer.weights.deltas, self.hiddenLayer.bias.deltas, self.outputLayer.weights.deltas, self.outputLayer.bias.deltas]:
            np.clip(dparam, -self.gradientClipSize, self.gradientClipSize, out=dparam)

        return gradInputState, dhiddenNext
