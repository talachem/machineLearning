import numpy as np
from numpy.typing import NDArray
from .layer import Layer
from .weights import Weights


def checkDims(input: NDArray, adjacencyMatrix: NDArray) -> None:
    """
    checks input shape/dimension for graph layer
    """

    # Check that the input array has at least 2 dimensions
    assert input.ndim >= 2, f"Input array should have at least 2 dimensions, got {input.ndim}"

    # Check that the adjacency matrix is 2-dimensional
    assert adjacencyMatrix.ndim == 2, f"Adjacency matrix should be 2-dimensional, got {adjacencyMatrix.ndim}"

    # Check that the adjacency matrix is square
    assert adjacencyMatrix.shape[0] == adjacencyMatrix.shape[1], f"Adjacency matrix should be square, got shape {adjacencyMatrix.shape}"

    # Check that the number of nodes in the input matches the number of nodes in the adjacency matrix
    assert input.shape[0] == adjacencyMatrix.shape[0], f"Number of nodes in input ({input.shape[0]}) should match number of nodes in adjacency matrix ({adjacencyMatrix.shape[0]})"


def diagEmbed(input: NDArray) -> NDArray:
    """
    Returns a diagonal embedding of the input array.
    Given a batch of vectors, diagonalizes each vector
    across the whole batch.
    """
    batchSize, dataSize = input.shape
    # Create an identity matrix of shape (dataSize, dataSize)
    identity = np.eye(dataSize)
    # Scale the identity matrix by the elements of the input matrix
    # and stack them along a new dimension
    diags = np.array([identity * row for row in input])
    # Return the batch of diagonal matrices
    return diags


def getDegree(adjacencyMatrix: NDArray) -> NDArray:
    """
    Returns the degree matrix of the input adjacency matrix.
    """
    epsilon = 1e-10
    return diagEmbed(np.power(adjacencyMatrix.sum(axis=-1) + epsilon, -0.5))


def getLaplacian(adjacencyMatrix: NDArray) -> NDArray:
    """
    Returns the Laplacian matrix of the input adjacency matrix.
    """
    degreeMatrix = getDegree(adjacencyMatrix)
    return np.matmul(np.matmul(degreeMatrix, adjacencyMatrix), degreeMatrix)


class GraphConvolution(Layer,):
    """
    Implementation of a graph convolution layer.
    """
    __slots__ = ['inputSize', 'outputSize', 'loops', 'laplacian', 'input', 'weights', 'bias']

    def __init__(self, inputSize: int, outputSize: int, weights: NDArray | None = None, bias: NDArray | None = None, selfLoops: bool = False) -> None:
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weights = Weights((inputSize, outputSize), values=weights)
        self.bias = Weights((1, outputSize), values=bias)
        self.loops = selfLoops

    def params(self) -> list[Weights]:
        """
        returns weights and bias in a python list, called by optimizers
        """
        return [self.weights, self.bias]

    def forward(self, input: NDArray, adjacencyMatrix: NDArray) -> tuple[NDArray, NDArray]:
        """
        Performs the forward pass of the graph convolution layer.
        """
        checkDims(input, adjacencyMatrix)

        # Reshape the input to a 2D array.
        self.input = input.reshape(-1, self.inputSize).squeeze()

        # Compute the output of the layer.
        output = np.matmul(self.input, self.weights.values) + self.bias.values

        # Add self-loops to the adjacency matrix if needed.
        if self.loops:
            adjacencyMatrix += np.eye(adjacencyMatrix.shape[0])

        # Compute the Laplacian matrix of the graph.
        self.laplacian = getLaplacian(adjacencyMatrix)

        # Perform the graph convolution operation.
        output = np.matmul(self.laplacian, output)

        # Return both the output and the adjacency matrix.
        # I return two things, to bring it into line with
        # RNN layer, where I have two inputs and two outputs
        return output, adjacencyMatrix

    def backward(self, gradient: NDArray) -> NDArray:
        """
        Performs backpropagation through the GraphConvolution layer.
        """
        # Compute the gradient of the loss with respect to the weights and bias.
        self.weights.deltas = np.matmul(self.input.T, gradient)
        self.bias.deltas = np.sum(gradient, axis=0, keepdims=True)

        # Compute the gradient of the loss with respect to the input.
        # Since the output of the layer is computed as y = Lx, where L is the
        # Laplacian matrix, the gradient of the loss with respect to the input
        # can be computed as L^T(y * weights^T), where ^T denotes the transpose
        # operator and * denotes element-wise multiplication.
        gradient = np.matmul(gradient, self.weights.values.T)
        gradient = np.matmul(self.laplacian.T, gradient)

        # Reshape the gradient to match the input shape.
        return gradient.reshape((-1, self.inputSize))
