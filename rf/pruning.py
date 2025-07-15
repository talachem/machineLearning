import numpy as np
from abc import ABC, abstractmethod

from numpy.typing import NDArray
from .decisionTree import DecisionTree


class Pruning(ABC):
    """
    Base class for pruning methods.
    """
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def __call__(self, tree: DecisionTree, evalData: NDArray, evalTargets: NDArray) -> None:
        self.prune(tree, evalData, evalTargets)

    @abstractmethod
    def prune(self, tree: DecisionTree, evalData: NDArray, evalTargets: NDArray) -> None:
        """
        Prune a decision tree.
        """
        pass


class ReducedError(Pruning):
    """
    Reduced Error Pruning.
    """
    def __init__(self) -> None:
        super().__init__()

    def prune(self, tree: DecisionTree, evalData: NDArray, evalTargets: NDArray) -> None:
        predictions = tree.eval(evalData)
        initialAccuracy = np.mean(predictions == evalTargets)

        # Traverse the tree in reverse (from leaves to root)
        for node in tree.breadthLast():
            if not node.hasChildren:  # Skip leaf nodes
                continue

            # Temporarily remove children
            initialValues = node.values
            initialLeft, initialRight = node.left, node.right
            node.popChildren()

            # Check accuracy on validation set
            predictions = tree.eval(evalData)
            prunedAccuracy = np.mean(predictions == evalTargets)

            # If accuracy decreases, restore children
            if prunedAccuracy < initialAccuracy:
                node.hasChildren = True
                node.values = initialValues
                node.left, node.right = initialLeft, initialRight


class CostComplexity(Pruning):
    """
    Cost Complexity Pruning.
    """
    def __init__(self) -> None:
        super().__init__()

    def prune(self, tree: DecisionTree, evalData: NDArray, evalTargets: NDArray) -> None:
        # Initial complexity and fit
        initialComplexity = tree.countNodes()
        initialFit = np.sum((tree.eval(evalData) - evalTargets) ** 2)

        # Traverse the tree in reverse (from leaves to root)
        for node in tree.breadthLast():
            if not node.hasChildren:  # Skip leaf nodes
                continue

            # Temporarily remove children
            initialValues = node.values
            initialLeft, initialRight = node.left, node.right
            node.popChildren()

            # Compute complexity and fit of pruned tree
            prunedComplexity = tree.countNodes()
            prunedFit = np.sum((tree.eval(evalData) - evalTargets) ** 2)

            # If sum of complexity and fit is higher for pruned tree, restore children
            if prunedComplexity + prunedFit > initialComplexity + initialFit:
                node.hasChildren = True
                #node.values = initialValues
                node.left, node.right = initialLeft, initialRight


class PessimisticError(Pruning):
    """
    Pessimistic Error Pruning.
    """
    def __init__(self):
        super().__init__()

    def prune(self, tree: DecisionTree, evalData: NDArray, evalTargets: NDArray) -> None:
        # Traverse the tree in reverse breadth-first order
        for node in tree.breadthLast():
            if not node.hasChildren:  # Skip leaf nodes
                continue

            # Calculate the estimated error rate of the node and its children
            nodeError = self.estimateError(node, evalData, evalTargets)
            childrenError = sum(self.estimateError(child, evalData, evalTargets) for child in node.getChildren())

            # If the node's error rate is lower, prune its children
            if nodeError <= childrenError:
                node.popChildren()

    def estimateError(self, node, evalData: NDArray, evalTargets: NDArray) -> float:
        """
        Estimate the error of the node using the validation set
        """
        # Get the indices of the data that fall within this node
        indices = np.where(evalData[:, node.feature] <= node.threshold)[0]

        # Check if node is a leaf or not
        if not node.hasChildren:
            # This node's prediction is the most common target value among the data in this node
            prediction = np.argmax(np.bincount(evalTargets[indices].astype(int)))

            # Compute error as the proportion of incorrect predictions
            error = np.sum(evalTargets[indices] != prediction) / len(indices)
        else:
            # This node is not a leaf, so its error is the average error of its children
            error = np.mean([self.estimateError(child, evalData, evalTargets) for child in node.getChildren()])

        return error
