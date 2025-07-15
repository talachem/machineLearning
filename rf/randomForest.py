import numpy as np
from numpy.typing import ArrayLike, NDArray
import warnings

from machineLearning.data.dataLoader import DataSet, DataLoader
from .decisionTree import DecisionTree
from .voting import Voting
from .boosting import Boosting
from .impurityMeasure import ImpurityMeasure
from .leafFunction import LeafFunction
from .splitAlgorithm import SplitAlgorithm
from .featureSelection import FeatureSelection
from collections import namedtuple
from importlib import import_module


class RandomForest(object):
    """
    the random forest class
    it works like a list for tree
    """
    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.trees: list[DecisionTree] = []

        # some flags
        self._trained = False
        self._baked = False

        # forest components
        self.boosting = None
        self.voting = None

        # tree components
        self._impurityMeasure: ImpurityMeasure | None = None
        self._leafFunction: LeafFunction | None = None
        self._splitAlgorithm: SplitAlgorithm | None = None
        self._featureSelection: FeatureSelection | None = None

    @property
    def qualifiedName(self) -> tuple:
        return self.__class__.__module__, self.__class__.__name__

    def toDict(self) -> dict:
        saveDict = {}
        saveDict['trained'] = self._trained
        saveDict['baked'] = self._baked

        if self.boosting:
            saveDict['boosting'] = {}
            saveDict['boosting']['qualifiedName'] = self.boosting.qualifiedName
            saveDict['boosting']['dict'] = self.boosting.toDict()
        else:
            saveDict['boosting'] = None

        if self.voting:
            saveDict['voting'] = self.voting.name
            saveDict['votingsWeights'] = self.voting.weights.tolist()
        else:
            saveDict['voting'] = None

        saveDict['trees'] = {}
        for tree in self.trees:
            saveDict['trees'][tree.id] = tree.toDict()
        return saveDict

    @classmethod
    def fromDict(cls, loadDict) -> 'RandomForest':
        instance = cls()  # replace YourClass with the actual name of your class
        instance._trained = loadDict['trained']
        instance._baked = loadDict['baked']

        if loadDict['voting'] != None:
            Module = import_module('machineLearning.rf.voting')  # dynamically import module
            Class = getattr(Module, loadDict['voting'])  # get class from imported module
            instance.voting = Class(loadDict['votingsWeights'])

        if loadDict['boosting'] != None:
            moduleName, className = loadDict['boosting']['qualifiedName']
            Module = import_module(moduleName)  # dynamically import module
            Class = getattr(Module, className)  # get class from imported module
            instance.boosting = Class().fromDict(loadDict['boosting']['dict'])

        for id in loadDict['trees']:
            tree = DecisionTree()
            tree = tree.fromDict(loadDict['trees'][id])
            instance.trees.append(tree)
        return instance

    def append(self, tree: DecisionTree) -> None:
        """
        append trees to the class
        if tree components have been set in the forest, the trees components will be over written
        """
        assert isinstance(tree, DecisionTree), 'only Decision Trees are allowed'
        if self._impurityMeasure:
            tree.setComponent(self._impurityMeasure)
        if self._leafFunction:
            tree.setComponent(self._leafFunction)
        if self._splitAlgorithm:
            tree.setComponent(self._splitAlgorithm)
        if self._featureSelection:
            tree.setComponent(self._featureSelection)
        self.trees.append(tree)

    def setComponent(self, component: Voting | Boosting) -> None:
        """
        setting different component types for forest and trees
        """
        if isinstance(component, Voting):
            self.voting = component
            if self.boosting == True:
                warnings.warn("Using Voting and Boosting togther works, but doesn't make much sense.'", UserWarning)

        elif isinstance(component, Boosting):
            self.boosting = component
            if self.voting == True:
                warnings.warn("Using Voting and Boosting togther works, but doesn't make much sense.'", UserWarning)

        elif isinstance(component, LeafFunction):
            self._leafFunction = component
            for tree in self.trees:
                tree.setComponent(component)
        elif isinstance(component, ImpurityMeasure):
            self._impurityMeasure = component
            for tree in self.trees:
                tree.setComponent(component)
        elif isinstance(component, SplitAlgorithm):
            self._splitAlgorithm = component
            for tree in self.trees:
                tree.setComponent(component)
        elif isinstance(component, FeatureSelection):
            self._featureSelection = component
            for tree in self.trees:
                tree.setComponent(component)
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

        for i, (strap, tree) in enumerate(zip(data.trainingSamples(self.numTrees), self.trees)):
            # skip already trained trees
            if tree._trained is True:
                continue

            # boosting
            if self.boosting:
                self.boosting.train(tree, strap)
            else:
                # fit a decision tree model to the current sample
                tree.train(strap)

        self._trained = True

    def eval(self, data: ArrayLike) -> NDArray:
        """
        predict from the ensemble
        """
        if self._trained is False:
            raise Exception('The forest must be trained before it can make predictions.')

        if self.voting == False and self.boosting == False:
            warnings.warn('None voting or boosting was set, hence each individual tree prediction will be returned.')

        # check is boosting has been set
        if self.boosting:
            predictions = self.boosting.eval(self.trees, data)
        else:
            # loop through each fitted model
            predictions = []
            for tree in self.trees:
                # make predictions on the input data
                pred = tree.eval(data)
                # append predictions to storage list
                predictions.append(pred.reshape(-1,1))

            # compute the ensemble prediction
            predictions = np.concatenate(predictions, axis=1)

        if self.voting:
            prediction = self.voting(predictions)

            # return the prediction
            return prediction

        return predictions

    def bake(self) -> None:
        for tree in self.trees:
            tree.bake()

        self._baked = True

    def __str__(self) -> str:
        """
        used for printing the forest in a human readable manner
        """
        treeStrings = [str(tree) for tree in self.trees]

        componentString = f'voting: {self.voting.name if self.voting else None}, booster: {self.boosting.name if self.boosting else None}\n'
        printString = ' forest '.center(len(componentString), '━')
        printString += '\n'
        printString += componentString
        printString += '\n'
        for i, treeString in enumerate(treeStrings):
            printString += treeString
            if i < len(self.trees) + 1:
                printString += '\n\n'
        return printString

    @property
    def featureImportance(self) -> NDArray:
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

    def __getitem__(self, index: int) -> DecisionTree:
        return self.trees[index]

    def __setitem__(self, index: int, tree: DecisionTree) -> None:
        self.trees[index] = tree
