from typing import Generator
import numpy as np
from typing import Iterator
from numpy._typing import NDArray
from numpy.core.numeric import ones_like
#from numpy.lib.arraysetops import unique


def getSplitIndices(dataAmount: int, split: int) -> NDArray:
    """
    This function is used for splitting datasets into training and validation sets.
    It takes the size and the split  fraction as argument and returns the indices at which to split.
    """
    splitSize = dataAmount // split
    residual = dataAmount % split
    if residual == 0:
        return np.arange(0, dataAmount+splitSize, splitSize)
    else:
        indices = np.arange(0, dataAmount, splitSize)
        indices[-split:] += np.clip(np.arange(0,split,1) - 1, 0, split)
        return indices


class DataSet(object):
    """
    This class handles data, its labels etc. All attributes will be frozen in order to prevent the user from manipulating
    the data accidentally.
    """
    __slots__ = ['data', 'adjacency', 'labels', 'uniques', 'targets', 'weights', 'classWeights']

    def __init__(self, data: NDArray, *, adjacency: NDArray | None = None, labels: NDArray | None = None, targets: NDArray | None = None, weights: NDArray | None = None, classWeights: NDArray | None = None) -> None:
        self.data = data
        #self.data.setflags(write=False) # making the data array immutable

        # setting up adjacency matrix
        self.adjacency = adjacency
        #if adjacency is not None:
        #    self.adjacency.setflags(write=False) # making the adjacency array immutable

        # setting up labels (should be integers)
        self.labels = labels
        self.uniques = np.unique(self.labels, axis=0) if not self.labels is None else None # number of different dataset in this data set

        #if labels is not None:
            #self.labels.setflags(write=False) # making the labels array immutable
            #self.uniques.setflags(write=False) # making the unique labels array immutable

        # setting up targets (should be floats)
        self.targets = targets
        #if targets is not None:
        #    self.targets.setflags(write=False) # making the targets array immutable

        # setting up class weights
        # used for rebalancing unbalanced data sets
        self.classWeights = classWeights if not classWeights is None else np.ones_like(self.uniques) if not self.uniques is None else None

        # setting up instance weights
        # weights should remain adjustable
        self.weights = weights if not weights is None else np.ones_like(self.data[:,0])

    def __getitem__(self, index: int | slice | NDArray) -> 'DataSet':
        """
        Return a new DataSet instance representing a subset of the original data.
        """
        # Create subsets of each attribute based on the index
        data_subset = self.data[index]

        adjacency_subset = self.adjacency[index] if self.adjacency is not None else None
        labels_subset = self.labels[index] if self.labels is not None else None
        targets_subset = self.targets[index] if self.targets is not None else None
        weights_subset = self.weights[index] if self.weights is not None else None
        # For classWeights, it usually applies to the entire dataset and might not need slicing

        # Create a new DataSet instance with the subsets
        subset = DataSet(data=data_subset, adjacency=adjacency_subset, labels=labels_subset,
                            targets=targets_subset, weights=weights_subset, classWeights=self.classWeights)

        return subset

    def selectFeatures(self, featureIndices: int) -> 'DataSet':
        """
        this method is here because of decision trees, they need to select
        a specific feature from a data set.
        """
        return DataSet(data=self.data[:,featureIndices], adjacency=self.adjacency, labels=self.labels,
                            targets=self.targets, weights=self.weights, classWeights=self.classWeights)

    def splitDataSet(self, splitIndices: NDArray | int | slice) -> 'DataSet':
        """
        this method is here because of decision trees, they need to split
        data sets on specific indices.
        """
        return DataSet(data = self.data[splitIndices],
            adjacency = self.adjacency[splitIndices] if not self.adjacency is None else None,
            labels = self.labels[splitIndices] if not self.labels is None else None,
            targets = self.targets[splitIndices] if not self.targets is None else None,
            weights = self.weights[splitIndices] if not self.weights is None else None,
            classWeights = self.classWeights if not self.classWeights is None else None)

    def __len__(self) -> int:
        """
        This allows the use of len(...) on this class.
        """
        return len(self.data)

    def __str__(self) -> str:
        """
        This method is not implemented.
        """
        return 'not implemented'

    def __iter__(self) -> Iterator:
        """
        this is included to enable max/min functions
        """
        return iter(self.data)

    def __contains__(self, key: str) -> bool:
        """
        Check if a given attribute exists and is not None in the DataSet.
        """
        return hasattr(self, key) and getattr(self, key) is not None

    @property
    def shape(self) -> tuple:
        return self.data.shape


class DataLoader():
    """
    This class handles iterating of datasets. Since dataset has only frozen attributes, one needs to use indices for
    shuffling the dataset. This class uses indices for creating an iterable.
    """
    __slots__ = ['dataSet', 'batchSize', '_batchSize', '_index', 'shuffle', 'kFold', '_order', '_indices', '_lengthes', '_mode', '_selection', '_lastElement', 'numTrainingSamples', '_bootstrapIndices', '_currentBoostrapSample']

    def __init__(self, dataSet: DataSet, batchSize: int, *, shuffle: bool = False, kFold: int = 1, useBootstrapping: bool = False, numTrainingSamples: int = 1) -> None:
        if not isinstance(dataSet, DataSet):
            raise TypeError(f'dataSet is not of type DataSet, it is {type(dataSet)}')

        self.dataSet = dataSet
        self.batchSize = batchSize # user controlled batchSize
        self._batchSize = 0 # this parameter is set automatically
        self._index = 0
        self.shuffle = shuffle

        if kFold > 1 and (numTrainingSamples > 1 and useBootstrapping == True):
            raise ValueError("cannot enable k-folding and bootsraping at the same time")

        self.kFold = kFold
        self.numTrainingSamples = numTrainingSamples

        # setting up kfolding
        self._mode = '' if kFold > 1 else 'simple'
        self._order = list(range(0, kFold, 1))
        indices = getSplitIndices(len(dataSet), kFold)
        self._indices = [np.arange(i,j,1) for i,j in zip(indices, indices[1:])]
        self._lengthes = [j-i for i,j in zip(indices, indices[1:])]

        if (numTrainingSamples > 1) and (useBootstrapping == True):
            self._mode = 'bootstrap'
            self._bootstrapIndices = [np.random.choice(len(self.dataSet), len(self.dataSet), replace=True) for _ in range(numTrainingSamples)]
        else:
            self._bootstrapIndices = []

        self._currentBoostrapSample = 0

        if not self._order and not self._bootstrapIndices:
            self._mode = 'simple'

        self._selection = None # used for storing indices that will be iterated over
        self._lastElement = 0 # total length of _selection, used for stopping iteration

    def __iter__(self) -> Iterator:
        """
        This gets called at the start of iterations, this allows us to
        manipulate how we iterate over any instance of this class.
        """
        if self._mode == 'simple':
            self._selection = self._indices[0]
            self._batchSize = self.batchSize
        elif self._mode == 'train':
            self._selection = np.concatenate([self._indices[index] for index in self._order[0:-1]])
            self._batchSize = self.batchSize
        elif self._mode == 'eval':
            self._selection = self._indices[self._order[-1]]
            self._batchSize = 2 * self.batchSize
        elif self._mode == 'bootstrap':
            if self._currentBoostrapSample < self.numTrainingSamples:
                self._selection = self._bootstrapIndices[self._currentBoostrapSample]
                self._batchSize = self.batchSize
                self._lastElement = len(self._selection) - 1
                self._index = -self.batchSize
                # Optionally shuffle within each bootstrap sample
                if self.shuffle:
                    np.random.shuffle(self._selection)
                self._currentBoostrapSample += 1
            else:
                raise StopIteration
        elif self._mode == '':
            raise ValueError('no mode has been set')
        else:
            raise ValueError('not a valid mode')

        # setting the last element, in order to know when to stop iterating
        self._lastElement = len(self._selection) - 1

        # the starting index, gets incrementally increased by batchsize in each loop
        self._index = -self.batchSize

        # shuffle the selection of indices, used for training
        if self.shuffle is True:
            np.random.shuffle(self._selection)
        return self

    def __next__(self) -> DataSet:
        """
        This gets called at every step of an iteration, this allows
        us to manipulate how we iterate over an instance of this class.
        This enables us to iterate using batch sizes.
        """
        if self._index + self._batchSize < self._lastElement:
            self._index += self._batchSize
            return self.dataSet[self._selection[self._index:self._index+self._batchSize]]
        else:
            raise StopIteration

    def __len__(self) -> int:
        """
        The total number of iterations and iterations of k-fold datasets. This is used to know in advance how long a loop is.
        """
        if self._mode == 'train' or self._mode == 'eval':
            trainIters = np.ceil(np.sum([self._lengthes[i] for i in self._order[:-1]]) / self.batchSize)
            evalIters = np.ceil(self._lengthes[self._order[-1]] / (2 * self.batchSize))
            totalIters = int(trainIters) + int(evalIters)
            addition = 1 if self.kFold % 2 == 0 else 0
            return totalIters - addition
        else:
            return int(np.ceil(len(self.dataSet) / self.batchSize))

    def fold(self) -> None:
        """
        This cycles through the indices used during iteration.
        """
        if len(self._order) > 1:
            self._order.append(self._order[0])
            self._order = self._order[1:]
        else:
            raise ValueError('There is only one dataset, nothing to fold.')

    def train(self) -> None:
        """
        Sets the class to training mode. In this mode, all
        index groups but the last will be combined during training mode.
        """
        if len(self._order) > 1:
            self._mode = 'train'
        else:
            raise ValueError('There is only one dataset, thus only pure evaluation.')

    def eval(self) -> None:
        """
        Sets the class to evaluation mode. In this mode, only the last
        index group will be used during evaluation mode.
        """
        if len(self._order) > 1:
            self._mode = 'eval'
        else:
            raise ValueError('There is only one dataset, thus only pure evaluation.')

    def trainingSamples(self, numTrainingSamples: int | None = None) -> Generator:
        """
        Yields entire bootstrap samples, one at a time.
        This is intended for use cases like training decision trees,
        where each tree is trained on a full bootstrap sample.
        """
        numTrainingSamples = self.numTrainingSamples if numTrainingSamples is None else numTrainingSamples
        if self._mode != 'bootstrap':
            for _ in range(numTrainingSamples):
                yield self.dataSet
        else:
            for bootstrap_index in self._bootstrapIndices:
                yield self.dataSet[bootstrap_index]

    @property
    def shape(self) -> tuple:
        return self.dataSet.shape

    @property
    def data(self) -> NDArray:
        return self.dataSet.data

    @property
    def targets(self) -> NDArray | None:
        return self.dataSet.targets

    @property
    def labels(self) -> NDArray | None:
        return self.dataSet.labels

    @property
    def adjacency(self) -> NDArray | None:
        return self.dataSet.adjacency

    @property
    def weights(self) -> NDArray | None:
        return self.dataSet.weights

    @property
    def classWeights(self) -> NDArray | None:
        return self.dataSet.classWeights
