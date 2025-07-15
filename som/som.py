import numpy as np
from numpy.typing import ArrayLike
from ..data.dataLoader import DataLoader, DataSet
from ..utility.progressbar import Progressbar
from .neighborhood import NeighborhoodFunction
from ..nn.scheduler import Scheduler
from importlib import import_module
from ..metric import Observables
from .error import Topological, Quantazation
from .topology import Topology


class SOM(object):
    """
    Main code for SOM
    """
    def __init__(self, learningRate: float, gridSteps: int = 1, decreaseEvery: int = None, growth: bool = False) -> None:
        self.name = self.__class__.__name__

        self.topology = None
        self.gridSteps = gridSteps
        self.decreaseEvery = decreaseEvery
        self.growth = growth
        self._initedWeights = False

        # this holds counts per neuron per class
        self.counts = []

        # parameters for training
        self.learningRate = learningRate
        self._scheduler = None

        # error estimation for training
        self._topoError = Topological()
        self._quantError = Quantazation()

        # neighborhood
        self.neighborhoodFunc = None

    @property
    def qualifiedName(self) -> tuple:
        """
        This is needed for saving the SOM as a json file
        """
        return self.__class__.__module__, self.__class__.__name__

    def toDict(self) -> dict:
        """
        This converts all values and features of the SOM into a dict
        Than it can be saved using built-in functions as json file
        """
        saveDict = {}
        saveDict['numFeatures'] = self.topology.numFeatures
        saveDict['gridSize'] = self.topology.gridSize
        saveDict['topology'] = self.topology.name
        saveDict['initedWeights'] = self._initedWeights

        saveDict['learningRate'] = self.learningRate
        saveDict['gridSteps'] = self.gridSteps
        saveDict['counts'] = []
        for item in self.counts:
            saveDict['counts'].append(item.tolist())
        if self._scheduler is not None:
            saveDict['scheduler'] = self._scheduler.name
        saveDict['neighborhoodFunc'] = self.neighborhoodFunc.name
        saveDict['scale'] = self.neighborhoodFunc.scale
        saveDict['weights'] = self.topology.weights.tolist()

        saveDict['movementCounts'] = self.topology._movementCounts.tolist()
        saveDict['directionCounts'] = self.topology._directionCounts.tolist()
        saveDict['minimalDistances'] = self.topology._minimalDistances.tolist()

        return saveDict

    @classmethod
    def fromDict(cls, loadDict: dict) -> object:
        """
        This instantiates a new SOM from a dict, which was loaded from a json file
        """
        instance = cls(loadDict['learningRate'], loadDict['gridSteps'])

        Module = import_module('machineLearning.som.neighborhood')  # dynamically import module
        Class = getattr(Module, loadDict['neighborhoodFunc'])  # get class from imported module
        instance.neighborhoodFunc = Class(loadDict['scale'])

        Module = import_module('machineLearning.som.topology')  # dynamically import module
        Class = getattr(Module, loadDict['topology'])  # get class from imported module
        instance.topology = Class(loadDict['gridSize'], loadDict['numFeatures'])
        instance._initedWeights = loadDict['initedWeights']

        instance.topology.weights = np.array(loadDict['weights'])
        instance.topology._movementCounts = np.array(loadDict['movementCounts'])
        instance.topology._directionCounts = np.array(loadDict['directionCounts'])
        instance.topology._minimalDistances = np.array(loadDict['minimalDistances'])
        for item in loadDict['counts']:
            instance.counts.append(item)

        return instance

    def setComponent(self, component: NeighborhoodFunction | Scheduler | Topology) -> None:
        """
        this allows setting a neighborhood functions and if the user
        wishes also a learning rate schdeuler
        """
        if isinstance(component, NeighborhoodFunction):
            self.neighborhoodFunc = component
        elif isinstance(component, Scheduler):
            self._scheduler = component
        elif isinstance(component, Topology):
            self.topology = component
        else:
            raise TypeError("the provided component is not a valid one")

    def initWeights(self, data: DataLoader | np.ndarray) -> None:
        # setting up the weight grid/neurons
        self.topology.initWeights(data)
        self._initedWeights = True

    def _findBestMatch(self, dataBatch: np.ndarray) -> np.ndarray:
        """
        Find the best matching unit (BMU) for each data point in the batch.
        """
        batchSize, numFeatures = dataBatch.shape

        # Repeat data points and tile weights
        repeatedData = np.repeat(dataBatch, self.topology.numNeurons, axis=0)
        tiledWeights = np.tile(self.topology.weights, (batchSize, 1))

        # Compute Euclidean distances between data points and weights
        norms = np.linalg.norm(tiledWeights - repeatedData, axis=1).reshape(batchSize, -1)

        # Partition distances so the two smallest distances are at the beginning
        partitioned = np.argpartition(norms, 2, axis=1)

        # Extract the indices of the best and second best matching units
        bestMatches, secondMatches = partitioned[:, 0], partitioned[:, 1]

        return bestMatches, secondMatches

    def _findNeighbors(self, bestMatches: ArrayLike) -> np.ndarray:
        """
        Given a list of best matching units (BMUs) on a grid, this function finds the indices of their neighboring units
        within a certain number of steps from them.
        """
        batchSize = len(bestMatches)
        uu, vv = np.unravel_index(bestMatches, self.topology.gridSize)
        indices = np.vstack((uu, vv)).T

        # broadcasting arrays to match each other
        tiledGrid = np.tile(self.topology.gridIndices, (batchSize, 1))
        repeatedIndices = np.repeat(indices, self.topology.numNeurons, axis=0)

        # distances understood as steps on a grid
        distances = np.sum(abs(tiledGrid - repeatedIndices), axis=-1).reshape(batchSize, -1)

        # the 'topological correction' is needed to find diagonal neighbors on a hexagonal grid
        inputVectors, neighbors = np.where(distances <= self.gridSteps + self.topology.stepCorrection)

        return inputVectors, neighbors

    def _adjustWeights(self, dataBatch: ArrayLike) -> None:
        """
        Adjusts the weights of a self-organizing map (SOM) given a batch of input vectors.
        """

        # Find the bestMatches for each input vector in the batch
        bestMatches, secondMatches = self._findBestMatch(dataBatch)

        # calculating and updating errors
        topoError = self._topoError(bestMatches, secondMatches)
        quantError = self._quantError(self.topology.weights[bestMatches], dataBatch)
        self.metrics.update('topologyError', topoError, len(dataBatch) * self.topology.numNeurons)
        self.metrics.update('quantazationError', quantError, len(dataBatch))

        # Find the indices of the neighboring neurons
        inputVectors, neighbors = self._findNeighbors(bestMatches)

        # Adjust the weights of the winning neuron and its neighbors
        deltaWeights = self.learningRate * (dataBatch - self.topology.weights[bestMatches])
        self.topology[bestMatches] += deltaWeights

        # Calculate the Gaussian neighborhood function for each neighbor
        neighborIndices = self.topology.gridIndices[neighbors]
        bestMatchIndices = self.topology.gridIndices[bestMatches[inputVectors]]
        topologicalDistances = np.linalg.norm(neighborIndices - bestMatchIndices, axis=-1)

        # finding the neighbors and determining weight changes
        neighborhoodWeights = self.neighborhoodFunc(topologicalDistances)
        neighborhoodWeights = np.repeat(neighborhoodWeights,self.topology.numFeatures).reshape(-1,self.topology.numFeatures)

        # updating neighborhood neurons
        deltaNeighbors = self.learningRate * neighborhoodWeights * (dataBatch[inputVectors] - self.topology.weights[neighbors])
        self.topology[neighbors] += deltaNeighbors

        # counting neuron movements
        self.topology.countMovements(bestMatches, neighbors, deltaWeights, deltaNeighbors)

    def train(self, data: DataLoader | np.ndarray, epochs: int, batchSize: int = None) -> None:
        """
        Trains the self-organizing map using the given data for the specified number of epochs and steps.
        """
        if self._initedWeights is False:
            self.initWeights(data)

        # testing input arguments
        self._checkInputDataType(data, batchSize)

        # converting numpy arrays into a DataLoader
        if isinstance(data, DataLoader) is False:
            data = DataSet(data)
            data = DataLoader(data, self._batchSize)

        # setting up training metrics
        self.metrics = Observables(epochs)
        self.metrics.addOberservable('topologyError', 'descending')
        self.metrics.addOberservable('quantazationError', 'descending')
        self.metrics.addOberservable('learningRate')
        self.metrics.addOberservable('gridSteps')

        # beginn training
        for i in range(epochs):
            length = len(data)
            bar = Progressbar(f'epoch {str(i+1).zfill(len(str(epochs)))}/{epochs}', length, 65) # setting up a progress bar

            # running over data batches
            for item in data:
                self._adjustWeights(item['data'])
                bar.step()

            # making a step with LR scheduler
            if self._scheduler is not None:
               self._scheduler.step()

            # growing the topology, this code is ugly
            if self.gridSteps > 1 and self.decreaseEvery:
                if i > 0 and i % self.decreaseEvery == 0:
                    self.gridSteps -= 1
            if i > 0 and i % 5 == 0 and self.growth:
                self.topology.grow()

            # updating and printing training metrics
            self.metrics.update('learningRate', self.learningRate)
            self.metrics.update('gridSteps', self.gridSteps)
            self.metrics.print()
            self.metrics.step()

    def _checkInputDataType(self, data, batchSize):
        """
        checking if data is given as a DataLoader or
        if data is given as numpy arrays together with a batchsize
        """
        if batchSize is None:
            assert type(data) == DataLoader, "if no batch size is provided, data should be data loader"
        elif batchSize is not None and type(data) == DataLoader:
            raise ValueError('batch size is already determined by data loader')
        else:
            self._batchSize = batchSize

    def eval(self, data: DataLoader | np.ndarray, labels: np.ndarray = None, batchSize: int = None, keepCounts: bool = True) -> None:
        """
        evaluates the class by counting the number of classes per neuron
        """

        # checking if weights were initilized
        if self._initedWeights is False:
            self.initWeights(data)

        # testing input arguments
        self._checkInputDataType(data, batchSize)

        # converting numpy arrays into a DataLoader
        if isinstance(data, DataLoader) is False:
            data = DataSet(data, labels=labels)
            data = DataLoader(data, self._batchSize)

        # setting up the counting arrays for evaluation
        uniqueLabels = data.dataSet.uniques
        self._counts = np.zeros((self.topology.numNeurons, len(uniqueLabels)))

        # running of data batches
        for item in data:
            self._eval(item['data'], item['labels'])

        if keepCounts is True:
            self.counts.append(self._counts)

    def _eval(self, data, labels):
        # Find the bestMatches for each input vector in the batch
        bestMatches = self._findBestMatch(data)

        # Use np.add.at() to handle repeated indices correctly
        if len(labels.shape) > 1:
            categoryLabels = np.argmax(labels, axis=1)
            np.add.at(self._counts, (bestMatches, categoryLabels), 1)
        else:
            np.add.at(self._counts, (bestMatches, labels), 1)

    @property
    def weightMatrix(self) -> np.ndarray:
        return self.topology.weightMatrix

    @property
    def uMatrix(self) -> np.ndarray:
        return self.topology.uMatrix

    @property
    def weights(self) -> np.ndarray:
        return self.topology.weights
