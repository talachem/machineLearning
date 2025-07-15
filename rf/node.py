import numpy as np
from numpy.typing import ArrayLike, NDArray


def saveObjectParameters(obj, saveDict, key):
    if isinstance(obj, np.ndarray):
        saveDict[key] = obj.tolist()
    elif isinstance (obj, (np.int8, np.int16, np.int32, np.int64)):
        saveDict[key] = int(obj)
    elif np.issubdtype(type(object), np.floating):
        saveDict[key] = float(obj)
    else:
        saveDict[key] = obj


class Node():
    """
    class to control tree nodes and leaves
    """
    count = 0 # class variable to track node count

    # initializer
    def __init__(self, level: int = 0, parent: int | None = None, isRoot: bool | None = False) -> None:
        # the feature and it's threshold this node handles
        self.threshold = None
        self.feature = None

        # left/right children
        self.left = None
        self.leftID = None
        self.right = None
        self.rightID = None

        # bool if this node has children
        self.hasChildren = False

        # if there are no children, the value this node holds
        self._rawValues = None
        self._bakedValues = None

        # metrics about this node
        self.isRoot = isRoot
        self.id = Node.count
        self.level = level
        self.parent = parent
        Node.count += 1

    @property
    def qualifiedName(self) -> tuple:
        return self.__class__.__module__, self.__class__.__name__

    def toDict(self) -> dict:
        saveDict = {}
        saveObjectParameters(self.threshold, saveDict, 'threshold')
        saveObjectParameters(self.feature, saveDict, 'feature')
        saveObjectParameters(self.leftID, saveDict, 'leftID')
        saveObjectParameters(self.rightID, saveDict, 'rightID')
        saveObjectParameters(self.id, saveDict, 'id')
        saveObjectParameters(self.isRoot, saveDict, 'isRoot')
        saveObjectParameters(self.parent, saveDict, 'parent')

        saveObjectParameters(self._rawValues, saveDict, 'rawValues')
        saveObjectParameters(self._bakedValues, saveDict, 'bakedValues')

        return saveDict

    def fromDict(self, loadDict: dict) -> None:
        self.threshold = loadDict['threshold']
        self.feature = loadDict['feature']
        self.id = loadDict['id']
        self.leftID = loadDict['leftID']
        self.rightID = loadDict['rightID']
        self.isRoot = loadDict['isRoot']
        self.parent = loadDict['parent']

        if type(loadDict['rawValues']) is list:
            self._rawValues = np.array(loadDict['rawValues'])
        else:
            self._rawValues = loadDict['rawValues']

        if type(loadDict['bakedValues']) is list:
            self._bakedValues = np.array(loadDict['bakedValues'])
        else:
            self._bakedValues = loadDict['bakedValues']

    def setValues(self, targets: NDArray) -> None:
        self._rawValues = targets

    @property
    def raws(self) -> NDArray:
        if self.hasChildren:
            left = self.left.raws
            right = self.right.raws
            return np.concatenate((left, right))
        return self._rawValues

    @property
    def values(self) -> NDArray:
        if self.hasChildren:
            left = self.left.values
            right = self.right.values
            return np.concatenate((left, right))
        if self._bakedValues:
            return self._bakedValues
        return self._rawValues

    @property
    def samples(self) -> int:
        return len(self.raws)

    def setParams(self, threshold: float, feature: float) -> None:
        """
        set the split, feature parameters for this node
        """
        self.threshold = threshold
        self.feature = feature

    def getParams(self) -> tuple:
        """
        get the split, feature parameters for this node
        """
        return self.threshold, self.feature

    def setChildren(self, left, right) -> None:
        """
        set the left/right children nodes for this current node
        """
        self.hasChildren = True
        self.left = left
        self.leftID = left.id
        self.right = right
        self.rightID = right.id

    def getChildren(self) -> tuple:
        """
        get the left and right child of this node
        """
        return self.left, self.right

    def popChildren(self) -> None:
        if self.hasChildren:
            self._rawValues = self.values
            self.hasChildren = False
            self.left = None
            self.right = None
        else:
            raise ValueError('this node has no children')

    def __str__(self) -> str:
        """
        used for printing the node in a human readable manner
        """
        if self.hasChildren:
            return f'feat: {self.feature} <= {self.threshold:.2f}, samples: {self.samples}'
        else:
            leafValue = self.raws

            if isinstance(leafValue[0], float):
                if all(np.mod(leafValue, 1) == 0):
                    values, counts = np.unique(leafValue, return_counts=True)
                    leafValue = values[counts.argmax()]
                else:
                    leafValue = np.mean(leafValue)
            else:
                values, counts = np.unique(leafValue, return_counts=True)
                leafValue = values[counts.argmax()]

            return f'value: {leafValue}'
