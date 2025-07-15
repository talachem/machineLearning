import numpy as np
from numpy.typing import NDArray
from .weights import Weights
from .rnn import RNN, checkDims


class LSTM(RNN):
    """
    An implementation of the LSTM layer.
    """
    __slots__ = ['inputSize', 'hiddenSize', 'input', 'hidden', 'cell']

    def __init__(self, inputSize: int, hiddenSize: int, weights: NDArray | None = None, bias: NDArray | None = None) -> None:
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        # Initialize weights and bias
        self.weights = Weights((inputSize + hiddenSize, 4 * hiddenSize), values=weights)
        self.bias = Weights((4 * hiddenSize,), values=bias)

        # Initialize hidden and cell states
        self.hidden = np.zeros((hiddenSize,))
        self.cell = np.zeros((hiddenSize,))

    def forward(self, input: NDArray, hiddenState: NDArray | None = None, cellState: NDArray | None = None) -> tuple[NDArray, NDArray, NDArray]:
        """
        forward pass of the LSTM layer
        """
        checkDims(input)

        self.input = input
        self.batchSize, self.seqLength, _ = input.shape

        # Initialize hidden and cell states if not provided
        if hiddenState is None:
            hiddenState = np.zeros((self.batchSize, self.seqLength, self.hiddenSize))
        if cellState is None:
            cellState = np.zeros((self.batchSize, self.seqLength, self.hiddenSize))

        # Initialize output array
        output = np.zeros((self.batchSize, self.seqLength, self.hiddenSize))

        for t in range(self.seqLength):
            combined = np.hstack((hiddenState[:, t, :], input[:, t, :]))
            gates = np.matmul(combined, self.weights.values) + self.bias.values

            # Compute the input, forget, and output gates
            inputGate, forgetGate, outputGate, hiddenGate = np.split(gates, 4)

            # Apply sigmoid activation function for input, forget, and output gates
            inputGate = 1 / (1 + np.exp(-inputGate))
            forgetGate = 1 / (1 + np.exp(-forgetGate))
            outputGate = 1 / (1 + np.exp(-outputGate))

            # Apply tanh activation function for the cell gate
            hiddenGate = np.tanh(hiddenGate)

            # Update the cell and hidden state
            cellState[:, t, :] = forgetGate * cellState[:, t, :] + inputGate * hiddenGate
            hiddenState[:, t, :] = outputGate * np.tanh(cellState[:, t, :])

        return output, hiddenState, cellState

    def backward(self, gradient: NDArray, hiddenGradient: NDArray | None = None, cellGradient: NDArray | None = None) -> tuple[NDArray, NDArray, NDArray]:
        """
        backward pass of the LSTM layer
        """
        gradInputState = np.zeros_like(self.input)
        dhiddenNext = np.zeros((self.batchSize, self.hiddenSize))
        dcellNext = np.zeros((self.batchSize, self.hiddenSize))
        dW = np.zeros_like(self.weights.values)
        db = np.zeros_like(self.bias.values)

        if hiddenGradient is not None:
            dhiddenNext += hiddenGradient
        if cellGradient is not None:
            dcellNext += cellGradient

        for t in reversed(range(self.seqLength)):
            # Compute the input, forget, and output gates
            inputGate, forgetGate, outputGate, hiddenGate = np.split(gradient[:, t, :], 4)

            # Partial derivative of loss w.r.t. output gate
            do = dhiddenNext * np.tanh(self.cell[:, t, :])
            do_input = do * outputGate * (1 - outputGate)

            # Partial derivative of loss w.r.t. cell state
            dc = dcellNext + dhiddenNext * outputGate * (1 - np.tanh(self.cell[:, t, :]) ** 2)
            dc_bar = dc * inputGate
            dc_bar_input = dc_bar * (1 - hiddenGate ** 2)

            # Partial derivative of loss w.r.t. input gate
            di = dc * hiddenGate
            di_input = di * inputGate * (1 - inputGate)

            # Partial derivative of loss w.r.t. forget gate
            df = dc * self.cell[:, t - 1, :]
            df_input = df * forgetGate * (1 - forgetGate)

            # Stacking the gradients
            dstacked = np.hstack((di_input, df_input, do_input, dc_bar_input))

            # Gradients with respect to weights and biases
            dW += np.matmul(np.hstack((self.input[:, t, :], self.hidden[:, t - 1, :])).T, dstacked)
            db += np.sum(dstacked, axis=0)

            # Gradients with respect to inputs
            gradInputState[:, t, :] = np.matmul(dstacked, self.weights.values[:self.inputSize].T)

            # Update for next timestep
            dhiddenNext = np.matmul(dstacked, self.weights.values[self.inputSize:].T)
            dcellNext = forgetGate * dc

        # Store the gradients
        self.weights.deltas = dW
        self.bias.deltas = db

        return gradInputState, dhiddenNext, dcellNext

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        printString = self.name
        printString += '    input size: ' + str(self.inputSize)
        printString += '    hidden size: ' + str(self.hiddenSize)
        return printString
