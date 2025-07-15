import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from ..data.dataLoader import DataLoader


def mapTo(values: np.ndarray, arange: list = [0,1]) -> np.ndarray:
    """
    this function maps any input values into a given range
    """
    assert len(arange) == 2, 'arange must be of length 2'
    assert arange[0] < arange[1], 'arange must start at a lower value than it ends'

    c, d = arange[0], arange[1]
    a, b = np.min(values), np.max(values)

    return c + ((d - c) / (b - a)) * (values - a)


class Topology(ABC):
    """
    An abstract base class that represents the topology of a Self-Organizing Map (SOM). The topology determines
    the arrangement of neurons in the SOM.
    """
    def __init__(self, gridSize: tuple[int, int], numFeatures: int):
        self.name = self.__class__.__name__
        self.stepCorrection = 0

        # basic parameters of the topology
        self.gridSize = list(gridSize)
        self.gridIndices = [] # used for visualizing the map as a scatter plot
        self.numFeatures = numFeatures
        self.numNeurons = np.prod(gridSize)

        # neighborhood relations
        self.neighborIndices = []
        self.neuronIndices = []

        # weights and neuron init
        self.weights = np.zeros((self.numNeurons, self.numFeatures)) # weights are in a flat list
        self.genIndices() # generating indices, used to find neighborhood relations

        # these arrays will be used assessing the topology
        self._movementCounts = np.zeros((self.numNeurons), dtype=int)
        self._directionCounts = np.zeros((self.numNeurons, self.numFeatures))
        self._travelDistance = np.zeros((self.numNeurons, self.numFeatures))
        self._minimalDistances = np.zeros((self.numNeurons, self.numFeatures))

    def initWeights(self, data: DataLoader | np.ndarray) -> None:
        """
        Initialize the weights of the neurons in the SOM using the given data.
        """
        # checking if data is given as numpy array or as a dataloader
        if isinstance(data, DataLoader):
            minValues = np.min(data.dataSet, axis=0)
            maxValues = np.max(data.dataSet, axis=0)
        else:
            minValues = np.min(data, axis=0)
            maxValues = np.max(data, axis=0)

        # Initialize weights within the range for each feature
        for i in range(self.numFeatures):
            self.weights[:, i] = np.random.uniform(minValues[i], maxValues[i], self.numNeurons)

    @property
    def weightMatrix(self) -> np.ndarray:
        """
        generates the weight matrix from neurons in order to plot it
        """
        return np.linalg.norm(self.weights,axis=-1).reshape(*self.gridSize)

    @property
    def uMatrix(self) -> np.ndarray:
        """
        generates the u-matrix from neurons in order to plot it
        """
        matrix = np.zeros(self.numNeurons)
        for index, weight in enumerate(self.weights):
            distances = np.linalg.norm(weight - self.weights[self.getNeighbors(index)], axis=-1)
            matrix[index] = np.mean(distances)

        return matrix.reshape(*self.gridSize)

    def __getitem__(self, index: int | ArrayLike) -> np.ndarray:
        """
        this allows getting values directly from the topology
        """
        if isinstance(index, (int | np.integer)):
            return self.weights[index]
        index = np.array(index)

        # here we check if an index is 2d and ravel it to match the shape of the topology
        if len(index.shape) == 2:
            index = np.ravel_multi_index(index, self.gridSize)
        return self.weights[index]

    def __setitem__(self, index: int | ArrayLike, value) -> None:
        """
        this allows setting values directly from the topology
        """
        if isinstance(index, int | np.integer):
            self.weights[index] = np.array(value)
            return
        index = np.array(index)

        # here we check if an index is 2d and ravel it to match the shape of the topology
        if len(index.shape) == 2:
            index = np.ravel_multi_index(index, self.gridSize)
        self.weights[index] = np.array(value)

    def __iter__(self):
        """
        this allows 'for ... in Topology'
        """
        return iter(self.weights)

    @abstractmethod
    def genIndices(self) -> None:
        raise NotImplementedError('this has not been implemented')

    def getNeighbors(self, index) -> np.ndarray:
        """
        Get the indices of the neurons that are neighbors to the neuron at the given index.
        """

        # checking if index is a single number
        if isinstance(index, (int, np.integer)):
            index = np.array([index])

        # broadcasting indices arrays
        tiledIndices = np.tile(self.neuronIndices, len(index))
        repeatedIndex = np.repeat(index, len(self.neuronIndices))

        # finding the equal values
        bools = (tiledIndices == repeatedIndex).reshape(len(index), len(self.neuronIndices))
        return self.neighborIndices[np.any(bools, axis=0)]

    def countMovements(self, bestMatches, neighbors, directions, neighborDirection) -> None:
        """
        Update the internal counts related to the movements and directions of the neurons.
        """
        np.add.at(self._movementCounts, bestMatches, 1)
        np.add.at(self._movementCounts, neighbors, 1)

        self._directionCounts[bestMatches] += np.sign(directions).astype('int')
        self._directionCounts[neighbors] += np.sign(neighborDirection).astype('int')

        #self._travelDistance[bestMatches] += directions
        #self._travelDistance[neighbors] += neighborDirection

    def grow(self):
        """
        Grow the topology of the SOM based on the internal counts of movements and directions.
        The growth happens in the direction where there is maximum movement and direction.

        This is still very much a toy/proof of concept model and needs lots of work
        """

        # storing current size/shape of the topology, these are lists because,
        # python allows jagged/ragged lists, while numpy doesn't, this makes
        # growing/inserting new neurons easier
        weightsShape = list((*self.gridSize, self.numFeatures))
        movementsShape = list(self.gridSize)

        # squeezing the movement counts into the range [0,1]
        movements = mapTo(self._movementCounts)

        # counting the directional changes
        directions = np.sum(abs(self._directionCounts),axis=1)
        directions = abs(mapTo(directions, [-1,0]))

        # bringing both together, the idea is that, if a neurons have high movement count,
        # but low directional change, these values should be low, I call it jittering... not an official term
        one, two = np.argpartition(movements + directions, -2)[-2:]

        # unraveling movement+direction counts
        x1, y1 = np.unravel_index(one, shape=(self.gridSize))
        x2, y2 = np.unravel_index(two, shape=(self.gridSize))

        # finding the biggest changes
        xDiff = abs(x1 - x2)
        yDiff = abs(y1 - y2)

        # adjusting new size of the topology
        if xDiff > yDiff == 0:
            weightsShape[1] += 1
            movementsShape[1] += 1
        else:
            weightsShape[0] += 1
            movementsShape[0] += 1

        # creating new arrays for metrics and weights
        newMovements = np.zeros(movementsShape, dtype=int)
        newWeights = np.zeros(weightsShape)
        newDirections = np.zeros(weightsShape, dtype=int)

        if xDiff > yDiff == 0:
            # adding neurons in the x-direction
            for i in range(newMovements.shape[0]):
                weightLine = self.weights.reshape(*self.gridSize, self.numFeatures)[i]
                movementLine = self._movementCounts.reshape(*self.gridSize)[i]
                directionLine = self._directionCounts.reshape(*self.gridSize, self.numFeatures)[i]

                mappedMovements = mapTo(movementLine)
                mappedDirections = abs(mapTo(np.sum(abs(directionLine)), [-1,0]))

                # finding the neuron with biggest jittering
                maxIndex = np.argmax(mappedMovements + mappedDirections)

                # checking if maxindex is at the edge
                if 0 < maxIndex < len(movementLine) - 1:
                    left = movementLine[maxIndex - 1]
                    right = movementLine[maxIndex + 1]
                    if left > right:
                        maxIndex -= 1

                # checking if maxindex is at the edge
                if maxIndex == 0:
                    maxIndex += 1

                # inserting newest neuron
                newMovementLine = np.insert(movementLine, maxIndex, 0)
                value = (weightLine[maxIndex-1] + weightLine[maxIndex])/2
                newWeightLine = np.insert(weightLine, maxIndex, value).reshape(-1,self.numFeatures)
                newDirectionLine = np.insert(directionLine, maxIndex, np.zeros(self.numFeatures,dtype=int)).reshape(-1,self.numFeatures)

                newMovements[i] = newMovementLine
                newWeights[i] = newWeightLine
                newDirections[i] = newDirectionLine
        else:
            # adding neurons in the y-direction
            for i in range(newMovements.shape[1]):
                weightLine = self.weights.reshape(*self.gridSize, self.numFeatures)[:,i]
                movementLine = self._movementCounts.reshape(*self.gridSize)[:,i]
                directionLine = self._directionCounts.reshape(*self.gridSize, self.numFeatures)[:,i]

                mappedMovements = mapTo(movementLine)
                mappedDirections = abs(mapTo(np.sum(abs(directionLine)), [-1,0]))

                # finding the neuron with biggest jittering
                maxIndex = np.argmax(mappedMovements + mappedDirections)

                # checking if maxindex is at the edge
                if 0 < maxIndex < len(movementLine) - 1:
                    left = movementLine[maxIndex - 1]
                    right = movementLine[maxIndex + 1]
                    if left > right:
                        maxIndex -= 1

                # checking if maxindex is at the edge
                if maxIndex == 0:
                    maxIndex += 1

                # inserting newest neuron
                newMovementLine = np.insert(movementLine, maxIndex, 0)
                value = (weightLine[maxIndex-1] + weightLine[maxIndex])/2
                newWeightLine = np.insert(weightLine, maxIndex, value).reshape(-1,self.numFeatures)
                newDirectionLine = np.insert(directionLine, maxIndex, np.zeros(self.numFeatures,dtype=int)).reshape(-1,self.numFeatures)

                newMovements[:,i] = newMovementLine
                newWeights[:,i] = newWeightLine
                newDirections[:,i] = newDirectionLine

        # overwriting old arrays with grown arrays
        self.gridSize = movementsShape
        self.numNeurons = np.prod(movementsShape)
        self._movementCounts = newMovements.reshape(self.numNeurons)
        self.weights = newWeights.reshape(self.numNeurons, self.numFeatures)
        self._directionCounts = newDirections.reshape(self.numNeurons, self.numFeatures)

        # updating grid indices after growing the topology
        self.genIndices()


class Rectangular(Topology):
    """
    A class used to represent a Rectangular topology of a Self Organizing Map (SOM).
    """
    def __init__(self, gridSize: tuple[int, int], numFeatures: int):
        super().__init__(gridSize, numFeatures)

    def genIndices(self) -> None:
        """
        Generates the indices for neurons and their respective neighbors in the rectangular grid.
        """
        self.gridIndices = []
        self.neighborIndices = []
        self.neuronIndices = []

        # Link the Neurons together in a grid pattern
        for index in range(self.numNeurons):
            position = [index // self.gridSize[0], index % self.gridSize[1]]
            self.gridIndices.append(position)

            neighbors = []
            if index >= self.gridSize[1]:
                neighbors.append(index-self.gridSize[1])  # up
            if index < self.numNeurons - self.gridSize[1]:
                neighbors.append(index+self.gridSize[1])  # down
            if index % self.gridSize[1] != 0:
                neighbors.append(index-1)  # left
            if (index+1) % self.gridSize[1] != 0:
                neighbors.append(index+1)  # right

            self.neighborIndices.extend(neighbors)
            self.neuronIndices.extend([index] * len(neighbors))

        self.gridIndices = np.array(self.gridIndices)
        self.neighborIndices = np.array(self.neighborIndices)
        self.neuronIndices = np.array(self.neuronIndices)


class Hexagonal(Topology):
    """
    A class used to represent a Hexagonal topology of a Self Organizing Map (SOM).
    """
    def __init__(self, gridSize: tuple[int, int], numFeatures: int):
        super().__init__(gridSize, numFeatures)
        self.stepCorrection = 0.5

    def genIndices(self) -> None:
        """
        Generates the indices for neurons and their respective neighbors in the hexagonal grid.
        """
        self.gridIndices = []
        self.neighborIndices = []
        self.neuronIndices = []

        # Link the Neurons together in a hexagonal pattern
        for index in range(self.numNeurons):
            position = [index // self.gridSize[0], index % self.gridSize[1]]
            if position[0] % 2 == 0:
               position[1] += 0.5
            self.gridIndices.append(position)

            neighbors = []
            if index >= self.gridSize[1]:
                neighbors.append(index-self.gridSize[1])  # up
            if index < self.numNeurons - self.gridSize[1]:
                neighbors.append(index+self.gridSize[1])  # down
            if index % self.gridSize[1] != 0:
                neighbors.append(index-1)  # left
            if (index+1) % self.gridSize[1] != 0:
                neighbors.append(index+1)  # right

            # diagonal neighbors (based on even or odd row)
            row = index // self.gridSize[1]
            if row % 2 == 0:  # even rows
                if index >= self.gridSize[1] and (index+1) % self.gridSize[1] != 0:
                    neighbors.append(index-self.gridSize[1]+1)  # upper right
                if index < self.numNeurons - self.gridSize[1] and (index+1) % self.gridSize[1] != 0:
                    neighbors.append(index+self.gridSize[1]+1)  # lower right
            else:  # odd rows
                if index >= self.gridSize[1] and index % self.gridSize[1] != 0:
                    neighbors.append(index-self.gridSize[1]-1)  # upper left
                if index < self.numNeurons - self.gridSize[1] and index % self.gridSize[1] != 0:
                    neighbors.append(index+self.gridSize[1]-1)  # lower left

            self.neighborIndices.extend(neighbors)
            self.neuronIndices.extend([index] * len(neighbors))

        self.gridIndices = np.array(self.gridIndices)
        self.neighborIndices = np.array(self.neighborIndices)