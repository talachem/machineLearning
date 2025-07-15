import numpy as np
from numpy.typing import ArrayLike, NDArray

from machineLearning.data.dataLoader import DataSet, DataLoader
from .decisionTree import DecisionTree
from .voting import Voting, Majority
from .boosting import Boosting, GradientBoosting
from collections import namedtuple
from importlib import import_module


class RandomForest(object):
    """
    the random forrest class
    it works like a list for tree
    """
    def __init__(self, *, retrainFirst: bool = False) -> None:
        self.name = self.__class__.__name__
        self.trees = []
        self._boosting = False
        self._booster = GradientBoosting()
        self.retrainFirst = retrainFirst

        # the voting algorithm to make the final decision
        self.voting = Majority()

        # some flags
        self._trained = False
        self._baked = False

    @property
    def qualifiedName(self) -> tuple:
        return self.__class__.__module__, self.__class__.__name__

    def toDict(self) -> dict:
        saveDict = {}
        saveDict['trained'] = self._trained
        saveDict['baked'] = self._baked
        saveDict['boosting'] = self._boosting
        saveDict['booster'] = self._booster.name
        saveDict['voting'] = self.voting.name
        saveDict['votingsWeights'] = self.voting.weights.tolist()
        saveDict['trees'] = {}
        for tree in self.trees:
            saveDict['trees'][tree.id] = tree.toDict()
        return saveDict

    @classmethod
    def fromDict(cls, loadDict) -> None:
        instance = cls()  # replace YourClass with the actual name of your class
        instance._trained = loadDict['trained']
        instance._baked = loadDict['baked']

        Module = import_module('machineLearning.rf.voting')  # dynamically import module
        Class = getattr(Module, loadDict['voting'])  # get class from imported module
        instance.voting = Class(loadDict['votingsWeights'])

        instance._boosting = loadDict['boosting']
        Module = import_module('machineLearning.rf.boosting')  # dynamically import module
        Class = getattr(Module, loadDict['booster'])  # get class from imported module
        instance._booster = Class()

        for id in loadDict['trees']:
            tree = DecisionTree()
            tree = tree.fromDict(loadDict['trees'][id])
            instance.trees.append(tree)
        return instance

    def append(self, tree: DecisionTree) -> None:
        """
        append trees to the class
        """
        assert issubclass(type(tree), DecisionTree), 'only Decision Trees are allowed'
        self.trees.append(tree)

    def setComponent(self, component: Voting | Boosting) -> None:
        """
        setting the voting algorithm
        """
        if isinstance(component, Voting):
            self.voting = component
        elif isinstance(component, Boosting):
            self._booster = component
            self._boosting = True
        else:
            raise TypeError("The given component is not a valid type")

    @property
    def numTrees(self) -> int:
        """
        reutrn the number of tree
        """
        return len(self.trees)

    def train(self, data: ArrayLike | DataSet | DataLoader, targets: NDArray | None = None, classWeights: NDArray | None = None, weights: NDArray | None = None) -> None:
        """
        protected function to train the ensemble
        """
        # If data is raw data (np.ndarray), convert it to DataSet first, then to DataLoader
        if not isinstance(data, (DataLoader, DataSet)):
            if targets is None:
                raise ValueError("When providing raw data as np.ndarray, 'targets' must also be provided.")
            # Assume data is raw, convert to DataSet
            data = DataSet(data, targets=targets, classWeights=classWeights)
        else:
            # Data is an instance of DataSet, check if targets were unnecessarily provided
            if targets is not None:
                raise ValueError("When providing data as a DataSet, 'targets' should not be provided separately.")
            if classWeights is not None:
                raise ValueError("When providing data as a DataSet, 'classWeights' should not be provided separately.")
            if weights is not None:
                raise ValueError("When providing data as a DataSet, 'weights' should not be provided separately.")

        # If data is DataSet, convert to DataLoader
        if isinstance(data, DataSet):
            data = DataLoader(data, batchSize=1, numTrainingSamples=self.numTrees)
        elif isinstance(data, DataLoader):
            # Assuming DataLoader is already correctly set up. Optionally, check or reconfigure DataLoader settings.
            if targets is not None or classWeights is not None:
                raise ValueError("When providing data as a DataLoader, 'targets' and 'classWeights' should not be provided.")
        else:
            raise ValueError("Data must be an instance of np.ndarray, DataSet, or DataLoader.")

        self.totalData, self.numFeatures = data.shape
        if self._boosting is False:
           self.retrainFirst = False

        if self.retrainFirst is True:
            treeDict = self.trees[0].toDict()
            tree = DecisionTree.fromDict(treeDict)
            self.trees.append(tree)

        for i, (strap, tree) in enumerate(zip(data.trainingSamples(self.numTrees), self.trees)):
            # skip already trained trees
            if tree._trained is True:
                continue

            # boosting
            if self._boosting:
                self._booster.train(tree, strap)
            else:
                # fit a decision tree model to the current sample
                tree.train(strap)
            if i == 0 and self.retrainFirst is True:
                print('·'*75)
        if self.retrainFirst is True:
            self.trees = self.trees[1:]
            DecisionTree.treeID -= 1

        self._trained = True

    def eval(self, data: ArrayLike) -> np.ndarray:
        """
        predict from the ensemble
        """
        if self._trained is False:
            raise Exception('The forrest must be trained before it can make predictions.')

        # loop through each fitted model
        predictions = []
        for tree in self.trees:
            # make predictions on the input data
            pred = tree.eval(data)
            # append predictions to storage list
            predictions.append(pred.reshape(-1,1))

        # compute the ensemble prediction
        predictions = np.concatenate(predictions, axis=1)
        prediction = self.voting(predictions)

        # return the prediction
        return prediction

    def bake(self) -> None:
        for tree in self.trees:
            tree.bake()

        self._baked = True

    def __str__(self) -> str:
        """
        used for printing the forrest in a human readable manner
        """
        treeStrings = [str(tree) for tree in self.trees]

        componentString = f'voting: {self.voting.name}, booster: {self._booster.name if self._boosting else None}\n'
        printString = ' forrest '.center(len(componentString), '━')
        printString += '\n'
        printString += componentString
        printString += '\n'
        for i, treeString in enumerate(treeStrings):
            printString += treeString
            if i < len(self.trees) + 1:
                printString += '\n\n'
        return printString

    @property
    def featureImportance(self) -> np.ndarray:
        """
        Computes the feature importance for each feature
        """
        featImportance = np.zeros(self.numFeatures)

        for tree in self.trees:
            featImportance += tree.featureImportance

        # not really sure I can do this, but wanted to clip out negative values
        featImportance = np.clip(featImportance, 0, None)
        return featImportance / sum(featImportance)

    def accuracy(self, data: np.ndarray, targets: np.ndarray) -> [namedtuple]:
        accuracy = namedtuple('Accuracy', 'name accuracy')
        accuarcies = []
        for tree in self.trees:
            score = accuracy(f'tree: {tree.id}', tree.accuracy(data, targets))
            accuarcies.append(score)
        return accuarcies

    def __len__(self) -> int:
        return len(self.trees)
