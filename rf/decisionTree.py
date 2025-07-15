import numpy as np
from uuid import uuid4
from numpy.typing import NDArray
from .node import Node
from .impurityMeasure import ImpurityMeasure, ODD
from .leafFunction import LeafFunction, AnomalyDetection
from .splitAlgorithm import SplitAlgorithm, RSA
from .featureSelection import FeatureSelection
from importlib import import_module
from inspect import signature
from ..utility.progressbar import Progressbar
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import warnings
from ..data.dataLoader import DataSet


def saveObjectParameters(obj, saveDict, key):
    saveDict[key] = {}
    saveDict[key]['name'] = obj.name
    params = list(signature(type(obj)).parameters)
    saveDict[key]['arguments'] = {}
    for param in params:
        if param == 'args' or param == 'kwargs':
            continue
        attr = getattr(obj, param)
        if isinstance(attr, (str, int, float, list, tuple)):
            saveDict[key]['arguments'][param] = attr
        elif isinstance(attr, np.ndarray):
            saveDict[key]['arguments'][param] = list(attr)


def loadObjectParameters(instance, attr_name, module_name, loadDict, key):
    Module = import_module(module_name)  # dynamically import module
    Class = getattr(Module, loadDict[key]['name'])  # get class from imported module
    setattr(instance, attr_name, Class(**loadDict[key]['arguments']))


class DecisionTree:
    """
    Implementation of a decision tree
    the user has to provide a splitting algorithm
    an impurity measure and a leaf function
    setting leaf values
    """
    treeNumber = 0 # counts the number of created trees

    # initializer
    def __init__(self, maxDepth: int | None = None, minSamplesSplit: int = 2) -> None:
        self.name = self.__class__.__name__
        self.root = None
        self.maxDepth = maxDepth
        if maxDepth is not None:
            self._maxNodes = (2 ** (maxDepth + 1)) - 1
        self.minSamplesSplit = minSamplesSplit
        self._trained = False
        self._baked = False

        # nodes and number of leafs and tracking actual depth
        self.numFeatures = None
        self.depth = 0
        self.leafs = 0

        # the components of the tree
        self._impurityMeasure = None
        self._leafFunction = None
        self._findSplit = None
        self._preSelect = None

        # counting the created trees
        self.id = str(uuid4())
        self.treeNum = DecisionTree.treeNumber
        DecisionTree.treeNumber += 1

    @property
    def qualifiedName(self) -> tuple:
        return self.__class__.__module__, self.__class__.__name__

    def toDict(self) -> dict:
        saveDict = {}
        saveDict['trained'] = self._trained
        saveDict['treeID'] = self.id
        saveDict['maxDepth'] = self.maxDepth
        self._maxNodes = (2 ** (self.maxDepth + 1)) - 1
        saveDict['depth'] = self.depth
        saveDict['minSamplesSplit'] = self.minSamplesSplit
        saveDict['leafFunction'] = self._leafFunction.name
        saveDict['baked'] = self._baked

        saveObjectParameters(self._impurityMeasure, saveDict, 'impurityMeasure')
        saveObjectParameters(self._findSplit, saveDict, 'splitAlgorithm')

        if self._preSelect is not None:
            saveObjectParameters(self._preSelect, saveDict, 'featureSelection')

        saveDict['nodes'] = {}
        for node in self.breadthFirst():
            saveDict['nodes'][node.id] = node.toDict()
        return saveDict

    @classmethod
    def fromDict(cls, loadDict: dict) -> object:
        instance = cls(loadDict['maxDepth'], loadDict['minSamplesSplit'])  # replace YourClass with the actual name of your class
        instance._trained = loadDict['trained']
        instance.id = loadDict['treeID']
        instance.depth = loadDict['depth']
        instance._baked = loadDict['baked']

        Module = import_module('machineLearning.rf.leafFunction')  # dynamically import module
        Class = getattr(Module, loadDict['leafFunction'])  # get class from imported module
        instance._leafFunction = Class()

        loadObjectParameters(instance, '_impurityMeasure', 'machineLearning.rf.impurityMeasure', loadDict, 'impurityMeasure')
        loadObjectParameters(instance, '_findSplit', 'machineLearning.rf.splitAlgorithm', loadDict, 'splitAlgorithm')
        instance._findSplit.setImpurityMeasure(instance._impurityMeasure)

        if 'featureSelection' in loadDict:
            loadObjectParameters(instance, '_preSelect', 'machineLearning.rf.featureSelection', loadDict, 'featureSelection')

        for key in loadDict['nodes']:
            if loadDict['nodes'][key]['isRoot']:
                instance.root = Node()
                instance.root.fromDict(loadDict['nodes'][key])
        if len(loadDict['nodes']) > 0:
            instance.setNodes(loadDict['nodes'], instance.root, instance.root.leftID, instance.root.rightID)
        return instance

    def setNodes(self, nodesDict: dict, node: Node, left: int, right: int) -> None:
        if left is None or right is None:
            return
        leftNode = Node()
        rightNode = Node()
        try:
            leftNode.fromDict(nodesDict[int(left)])
            rightNode.fromDict(nodesDict[int(right)])
        except KeyError:
            leftNode.fromDict(nodesDict[str(left)])
            rightNode.fromDict(nodesDict[str(right)])
        node.setChildren(leftNode, rightNode)
        self.setNodes(nodesDict, leftNode, leftNode.leftID, leftNode.rightID)
        self.setNodes(nodesDict, rightNode, rightNode.leftID, rightNode.rightID)

    def setComponent(self, component: ImpurityMeasure | LeafFunction | SplitAlgorithm | FeatureSelection) -> None:
        """
        setting components for the tree
        """
        if isinstance(component, ImpurityMeasure):
            self._impurityMeasure = component
            if self._findSplit is not None:
                self._findSplit.setImpurityMeasure(self._impurityMeasure)
        elif isinstance(component, LeafFunction):
            self._leafFunction = component
        elif isinstance(component, SplitAlgorithm):
            self._findSplit = component
            if self._impurityMeasure is not None:
                self._findSplit.setImpurityMeasure(self._impurityMeasure)
        elif isinstance(component, FeatureSelection):
            self._preSelect = component
        else:
            raise TypeError("The given component is not a valid type")

    def _grow(self, node: Node, level: int, data: DataSet) -> None:
        """
        this grows the tree in a recursive manner
        """
        if self.depth < level:
            self.depth = level
        # not a leaf node
        if self._continueGrowingTree(data, level):
            # find the best feature to split on
            if self._preSelect is not None:
                featureIndices = self._preSelect(data.data, data.targets)
            else:
                featureIndices = np.arange(0, data.shape[1])

            feature, threshold = self._findSplit(data.data[:,featureIndices], data.targets, weights=data.weights, classWeights=data.classWeights)

            if feature is None or threshold is None:
                # No valid split found, create a leaf node
                # this could cause problems
                node.setValues(data.targets)
                self.leafs += 1
                return

            # split the dataset based on the best feature
            leftSplit = data.data[:, feature] <= threshold
            rightSplit = data.data[:, feature] > threshold
            dataLeft = data.data[leftSplit]
            dataRight = data.data[rightSplit]
            if data.targets is not None:
                targetsLeft = data.targets[leftSplit]
                targetsRight = data.targets[rightSplit]
            else:
                targetsLeft = None
                targetsRight = None
            if data.labels is not None:
                labelsLeft = data.labels[leftSplit]
                labelsRight = data.labels[rightSplit]
            else:
                labelsLeft = None
                labelsRight = None
            if data.weights is not None:
                weightsLeft = data.weights[leftSplit]
                weightsRight = data.weights[rightSplit]
            else:
                weightsLeft = None
                weightsRight = None

            # check if the splits are empty
            if (len(dataLeft) == 0) or (len(dataRight) == 0):
                node.setValues(data.targets)
                self.leafs += 1
                return

            # set the current node's parameters
            node.setParams(threshold, feature)

            # declare child nodes
            leftNode = Node(level+1, node.id)
            self.bar.step()
            rightNode = Node(level+1, node.id)
            self.bar.step()
            node.setChildren(leftNode, rightNode)

            # investigate child nodes
            self._grow(leftNode, level+1, data=DataSet(dataLeft, targets=targetsLeft, classWeights=data.classWeights, weights=weightsLeft))
            self._grow(rightNode, level+1, data=DataSet(dataRight, targets=targetsRight, classWeights=data.classWeights, weights=weightsRight))
        # is a leaf node
        else:
            node.setValues(data.targets)
            self.leafs += 1

    def _continueGrowingTree(self, data: DataSet, level: int) -> bool:
        """
        Stop growing if we reached the maximum depth
        """
        if self.maxDepth is not None and level >= self.maxDepth:
            return False

        # here the tree is testing if targets or labels have been used
        # and to ensure that at least targets or labels are there
        # in case of OOD, neither are necesary
        # Stop growing if there are not enough samples to split
        if data.targets is not None:
            if len(data.targets) < self.minSamplesSplit:
                return False

            # Stop growing if all samples have the same target value
            if len(np.unique(data.targets)) == 1:
                return False

        # Stop growing if there are not enough samples to split
        if data.labels is not None:
            if len(data.labels) < self.minSamplesSplit:
                return False

            # Stop growing if all samples have the same target value
            if len(np.unique(data.labels)) == 1:
                return False

        return True

    def train(self, data: NDArray | DataSet, targets: NDArray | None = None, classWeights: NDArray | None = None, weights: NDArray | None = None) -> None:
        """
        training the tree
        """
        if not self._impurityMeasure or not self._findSplit:
            raise AttributeError('some (impurity or split algo) have not been set')

        # Check if data is not an instance of DataSet, then targets must be provided
        if not isinstance(data, DataSet):
            if targets is None:
                raise ValueError("When providing raw data as NDArray, 'targets' must also be provided.")
            data = DataSet(data, targets=targets, classWeights=classWeights, weights=targets)  # Convert to DataSet
        else:
            # Data is an instance of DataSet, check if targets were unnecessarily provided
            if targets is not None:
                raise ValueError("When providing data as a DataSet, 'targets' should not be provided separately.")
            if classWeights is not None:
                raise ValueError("When providing data as a DataSet, 'classWeights' should not be provided separately.")
            if weights is not None:
                raise ValueError("When providing data as a DataSet, 'weights' should not be provided separately.")

        # set the root node of the tree
        id = str(self.treeNum+1).zfill(len(str(self.treeNumber+1)))
        self.bar = Progressbar(f'tree {id}', self._maxNodes)
        self.root = Node(level=1, isRoot=True)
        self.bar.step()
        self.totalData, self.numFeatures = data.shape

        # build the tree
        self._grow(self.root, level=1, data=data)
        self._trained = True
        self.bar.finish()

    def eval(self, data: NDArray | DataSet) -> NDArray:
        """
        make predictions from the trained tree
        """
        if self._trained is False:
            raise Exception('The tree must be trained before it can make predictions.')

        if isinstance(data, DataSet) is False:
            data = DataSet(data=data)

        if not self._leafFunction:
            warnings.warn("Since no leaf function was defined, the raw leaf values are returned.")
            return self.raw(data.data)

        if not self._baked:
            # getting the raw leaf values
            nodes = []
            for r in range(len(data)):
                node = self._traverse(self.root, data.data[r])
                nodes.append(node)

            # iterating over raw predictions
            with ThreadPoolExecutor(max_workers=None) as executor:
                # Assuming 'rawPredictions' is a list of NDArray objects and you don't have 'nodes' yet.
                predictions = list(executor.map(self._leafFunction, nodes))
        else:
            predictions = [0] * len(data)

            # iterate through the rows of data
            for r in range(len(data)):
                node = self._traverse(self.root, data.data[r])
                predictions[r] = node._bakedValues

        return np.array(predictions)

    def raw(self, data: NDArray | DataSet) -> list:
        """
        make predictions from the trained tree
        """
        if self._trained is False:
            raise ValueError('must train the tree before it can make predictions')

        if isinstance(data, DataSet) is False:
            data = DataSet(data=data)

        predictions = [0] * len(data.data)

        # iterate through the rows of data
        for r in range(len(data)):
            node = self._traverse(self.root, data.data[r])
            predictions[r] = node._rawValues

        return predictions

    def bake(self) -> None:
        """
        baking the tree accelerators the prediction process
        """
        if self._trained is False:
            raise Exception('The tree must be trained before it can make predictions.')

        if not self._leafFunction:
            raise Exception("Since no leaf function was defined, the raw leaf values are returned.")

        for node in self.breadthFirst():
            if node._rawValues is not None:
                node._bakedValues = self._leafFunction(node)

        self._baked = True

    def _traverse(self, node: Node, dataPoint: NDArray) -> Node:
        """
        recursive function to traverse the (trained) tree
        """
        # check if we're in a leaf node?
        if node.hasChildren:
            # get parameters at the node
            threshold, feature = node.getParams()
            leftNode, rightNode = node.getChildren()
            # decide to go leftNode or rightNode?
            if (dataPoint[feature] <= threshold):
                return self._traverse(leftNode, dataPoint)
            else:
                return self._traverse(rightNode, dataPoint)
        else:
            return node

    def accuracy(self, data: NDArray | DataSet, targets: NDArray | None = None) -> float:
        if isinstance(data, DataSet) is False:
            data = DataSet(data=data, targets=targets)

        predictions = self.eval(data.data)
        result = np.sum(predictions == data.targets) / len(data)
        return result

    def breadthLast(self) -> list:
        q = deque()
        q.append(self.root)
        ans = []
        while q:
            node = q.popleft()
            if node is None:
                continue

            ans.insert(0, node)

            if node.right:
                q.append(node.right)

            if node.left:
                q.append(node.left)

        return ans

    def breadthFirst(self) -> list:
        q = deque()
        q.append(self.root)
        ans = []
        while q:
            node = q.popleft()
            if node is None:
                continue

            ans.append(node)

            if node.left:
                q.append(node.left)

            if node.right:
                q.append(node.right)

        return ans

    @property
    def nodes(self) -> list:
        return self.breadthFirst()

    def countNodes(self) -> int:
        return len(self.breadthFirst())

    @property
    def featureImportance(self) -> NDArray:
        """
        Computes the feature importance for each feature
        """
        featImportance = np.zeros(self.numFeatures)

        for node in self.breadthFirst():
            if node.left and node.right:
                nodeImpurity = (node.samples / self.totalData) * self._impurityMeasure(node.raws)
                leftImpurity = (node.left.samples / self.totalData) * self._impurityMeasure(node.left.raws)
                rightImpurity = (node.right.samples / self.totalData) * self._impurityMeasure(node.right.raws)
                impurityReduction = nodeImpurity - leftImpurity - rightImpurity
                featImportance[node.feature] += impurityReduction

        # not really sure I can do this, but wanted to clip out negative values
        featImportance = np.clip(featImportance, 0, None)
        return featImportance / sum(featImportance)

    def __len__(self) -> int:
        """
        number of leafs
        """
        return len(self.breadthFirst())

    def __str__(self) -> str:
        """
        used for printing the tree in a human readable manner
        """
        lines = self._buildTreeStructure(self.root, '', True, [])
        componentLine = f'\nsplit: {self._findSplit.name}, impurity: {self._impurityMeasure.name}, leaf: {self._leafFunction.name}, nodes: {self.countNodes()}\n'
        parameterLine = f'maxDepth: {self.maxDepth}, reached depth: {self.depth}, minSamplesSplit: {self.minSamplesSplit}\n'
        if self._trained is True:
            center = len(max(lines, key = len))
            center = max(center, len(componentLine), len(parameterLine)) + 1
        else:
            center = max(len(componentLine), len(parameterLine)) + 1

        treeID = str(self.treeNum+1).zfill(len(str(self.treeNumber)))
        firstLine = f' tree: {treeID}/{self.treeNumber} '.center(center, '—')
        firstLine += componentLine
        firstLine += parameterLine
        firstLine += '·' * center + '\n'
        return firstLine + '\n'.join(lines)[3:]

    def _buildTreeStructure(self, node: Node, prefix: str, isLastChild: bool, lines: list[str]) -> list[str]:
        """
        used by the __str__ method for printing the tree
        """
        if not node:
            return lines
        line = str(node)
        leftNode, rightNode = node.getChildren()
        if not leftNode and not rightNode:
            lines.append(prefix + ' └─╴' + line)
            return lines
        if isLastChild:
            lines.append(prefix + ' └─╴' + line)
            self._buildTreeStructure(leftNode, prefix + '    ', False, lines)
            self._buildTreeStructure(rightNode, prefix + '    ', True, lines)
        else:
            lines.append(prefix + ' ├─' + line)
            self._buildTreeStructure(leftNode, prefix + ' │  ', False, lines)
            self._buildTreeStructure(rightNode, prefix + ' │  ', True, lines)

        return lines
