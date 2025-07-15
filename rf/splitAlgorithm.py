import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from .impurityMeasure import ImpurityMeasure
from ..data.dataLoader import DataSet
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from copy import copy


class SplitAlgorithm(ABC):
    """
    this is a base class for implementing different algorithms
    for splitting features in data sets and finding thresholds
    """
    def __init__(self, featurePercentile: int | None = None, threads: int = 4, executor: str = 'thread') -> None:
        self.name = self.__class__.__name__
        self.featurePercentile = featurePercentile
        self.threads = threads
        self.executor = ProcessPoolExecutor if executor == 'process' else ThreadPoolExecutor
        self._maxInfoGain = None

    def __call__(self, data: NDArray, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> tuple[int | None, float | None]:
        """
        making it a bit more convenient to use this class
        """
        return self.findBestSplit(data, targets, weights=weights, classWeights=classWeights)

    def setImpurityMeasure(self, impurityMeasure: ImpurityMeasure) -> None:
        """
        the optimal split depends in the impurity measure
        that is why the algorithm needs to have one
        """
        if isinstance(impurityMeasure, ImpurityMeasure):
            self._impurity = impurityMeasure
        else:
            raise TypeError("impurity measure is not of class ImpurityMeasure")

    def getPercentiles(self, features: NDArray) -> NDArray:
        """
        Get percentiles of feature values
        """
        if self.featurePercentile is None:
            return features
        percentiles = np.linspace(0, 100, self.featurePercentile+2)[1:-1]  # Exclude 0 and 100
        return np.percentile(features, percentiles)

    @abstractmethod
    def _calculateInfoGain(self, feat: int, data: NDArray, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> tuple[float, int, float] | None:
        pass

    @abstractmethod
    def _goodGain(self, currentInfoGain: float, maxInfoGain: float) -> bool:
        pass

    def findBestSplit(self, data: NDArray, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> tuple[int | None, float | None]:
        maxInfoGain = copy(self._maxInfoGain)
        feature = None
        threshold = None

        # Create a thread pool with a maximum of 4 threads
        with self.executor(max_workers=self.threads) as executor:
            # Submit each feature to the thread pool for processing
            futures = [executor.submit(self._calculateInfoGain, feat, data, targets, weights=weights, classWeights=classWeights) for feat in range(data.shape[1])]

            for future in as_completed(futures):
                try:
                    currentInfoGain, currentFeature, currentThreshold = future.result()  # This will re-raise any exceptions that occurred in the thread
                except TypeError:
                    continue
                if self._goodGain(currentInfoGain, maxInfoGain):
                    maxInfoGain = currentInfoGain
                    feature = currentFeature
                    threshold = currentThreshold

        return feature, threshold


class CART(SplitAlgorithm):
    """
    CART (Classification and Regression Trees)
    this is the most robust algorithm
    also suitable for most problems
    """
    def __init__(self, featurePercentile: int | None = None, threads: int = 4, executor: str = 'thread') -> None:
        super().__init__(featurePercentile, threads, executor)
        self._maxInfoGain = np.inf

    def _calculateInfoGain(self, feat: int, data: NDArray, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> tuple[float, int, float] | None:
        line = data[:, feat]
        splits = self.getPercentiles(line)
        splits = np.unique(splits)
        allSplitsMask = line[:, None] <= splits
        leftCounts = np.sum(allSplitsMask, axis=0)
        rightCounts = len(data) - leftCounts
        validSplits = (leftCounts > 0) & (rightCounts > 0)

        if not np.any(validSplits):
            return

        leftTargets = [targets[splitMask] for splitMask in allSplitsMask.T]
        rightTargets = [targets[np.logical_not(splitMask)] for splitMask in allSplitsMask.T]

        if weights is not None:
            leftWeights = [weights[splitMask] for splitMask in allSplitsMask.T]
            rightWeights = [weights[np.logical_not(splitMask)] for splitMask in allSplitsMask.T]
        else:
            leftWeights = [None] * len(allSplitsMask)
            rightWeights = [None] * len(allSplitsMask)

        leftImpurity = np.array([(leftCounts[i] / len(targets)) * self._impurity(leftTargets[i], weights=leftWeights[i], classWeights=classWeights) for i in range(len(splits))])
        rightImpurity = np.array([(rightCounts[i] / len(targets)) * self._impurity(rightTargets[i], weights=rightWeights[i], classWeights=classWeights) for i in range(len(splits))])

        # collecting the impurity together
        infoGain = leftImpurity + rightImpurity
        infoGain = infoGain[validSplits]
        splits = splits[validSplits]

        minInfoGainIndex = np.argmin(infoGain)
        currentInfoGain = infoGain[minInfoGainIndex]

        return currentInfoGain, feat, splits[minInfoGainIndex]

    def _goodGain(self, currentInfoGain: float, maxInfoGain: float) -> bool:
        return currentInfoGain < maxInfoGain


class ID3(SplitAlgorithm):
    """
    ID3 (Iterative Dichotomiser 3)
    this is used for classification tasks
    less robust than CART
    """
    def __init__(self, featurePercentile: int | None = None, threads: int = 4, executor: str = 'thread') -> None:
        super().__init__(featurePercentile, threads, executor)
        self._maxInfoGain = -np.inf

    def _calculateInfoGain(self, feat: int, data: NDArray, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> tuple[float, int, float] | None:
        line = data[:, feat]
        splits = self.getPercentiles(line)
        splits = np.unique(splits)
        allSplitsMask = line[:, None] <= splits
        leftCounts = np.sum(allSplitsMask, axis=0)
        rightCounts = len(targets) - leftCounts
        validSplits = (leftCounts > 0) & (rightCounts > 0)

        if not np.any(validSplits):
            return

        leftTargets = [targets[splitMask] for splitMask in allSplitsMask.T]
        rightTargets = [targets[np.logical_not(splitMask)] for splitMask in allSplitsMask.T]

        if weights is not None:
            leftWeights = [weights[splitMask] for splitMask in allSplitsMask.T]
            rightWeights = [weights[np.logical_not(splitMask)] for splitMask in allSplitsMask.T]
        else:
            leftWeights = [None] * len(allSplitsMask)
            rightWeights = [None] * len(allSplitsMask)

        leftImpurity = np.array([(leftCounts[i] / len(targets)) * self._impurity(leftTargets[i], weights=leftWeights[i], classWeights=classWeights) for i in range(len(splits))])
        rightImpurity = np.array([(rightCounts[i] / len(targets)) * self._impurity(rightTargets[i], weights=rightWeights[i], classWeights=classWeights) for i in range(len(splits))])

        # calculating the left/right side contribution to impurity
        leftImpurityContrib = leftCounts / len(targets) * leftImpurity
        rightImpurityContrib = rightCounts / len(targets) * rightImpurity

        # Calculate information gain using the ID3 algorithm
        infoGain = self._impurity(targets) - leftImpurityContrib - rightImpurityContrib
        maxIndex = np.argmax(infoGain)

        return infoGain[maxIndex], feat, splits[maxIndex]

    def _goodGain(self, currentInfoGain: float, maxInfoGain: float) -> bool:
        return currentInfoGain > maxInfoGain


class C45(SplitAlgorithm):
    """
    C4.5 is an extended version of ID3
    """
    def __init__(self, featurePercentile: int | None = None, threads: int = 4, executor: str = 'thread', epsilon: float = 1e-10) -> None:
        super().__init__(featurePercentile, threads, executor)
        self.epsilon = epsilon # stability constant
        self._maxInfoGain = -np.inf

    def _calculateInfoGain(self, feat: int, data: NDArray, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> tuple[float, int, float] | None:
        line = data[:, feat]
        splits = self.getPercentiles(line)
        splits = np.unique(splits)
        allSplitsMask = line[:, None] <= splits
        leftCounts = np.sum(allSplitsMask, axis=0)
        rightCounts = len(targets) - leftCounts
        validSplits = (leftCounts > 0) & (rightCounts > 0)

        if not np.any(validSplits):
            return

        leftTargets = [targets[splitMask] for splitMask in allSplitsMask.T]
        rightTargets = [targets[np.logical_not(splitMask)] for splitMask in allSplitsMask.T]

        if weights is not None:
            leftWeights = [weights[splitMask] for splitMask in allSplitsMask.T]
            rightWeights = [weights[np.logical_not(splitMask)] for splitMask in allSplitsMask.T]
        else:
            leftWeights = [None] * len(allSplitsMask)
            rightWeights = [None] * len(allSplitsMask)

        leftImpurity = np.array([(leftCounts[i] / len(targets)) * self._impurity(leftTargets[i], weights=leftWeights[i], classWeights=classWeights) for i in range(len(splits))])
        rightImpurity = np.array([(rightCounts[i] / len(targets)) * self._impurity(rightTargets[i], weights=rightWeights[i], classWeights=classWeights) for i in range(len(splits))])

        # calculating the left/right side contribution to impurity
        leftImpurityContrib = leftCounts / len(targets) * leftImpurity
        rightImpurityContrib = rightCounts / len(targets) * rightImpurity

        # Calculate information gain using the C45 algorithm
        infoGain = self._impurity(targets) - leftImpurityContrib - rightImpurityContrib

        # Calculate split info
        left = (leftCounts / len(targets))
        right = (rightCounts / len(targets))
        splitInfo = - (left * np.log2(left + self.epsilon)) - (right * np.log2(right + self.epsilon))
        splitInfo[splitInfo == 0] += self.epsilon  # avoid dividing by zero

        # Calculate information gain ratio
        infoGainRatio = infoGain / splitInfo

        maxIndex = np.argmax(infoGainRatio)

        return infoGainRatio[maxIndex], feat, splits[maxIndex]

    def _goodGain(self, currentInfoGain: float, maxInfoGain: float) -> bool:
        return currentInfoGain > maxInfoGain


class RSA(SplitAlgorithm):
    """
    RSA - Random Split Algorithm
    Implementaion of ODD tree splitting algorithm
    This is the point where inheritince breaks down.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _calculateInfoGain(self, feat: int, data: NDArray, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> tuple[float, int, float] | None:
        # Randomly select a feature and a split value for that feature
        # This method simplifies to choosing a feature and then a random point as the split
        pass

    def _goodGain(self, currentInfoGain: float, maxInfoGain: float) -> None:
        # Since we're doing random splits, the concept of "good gain" might be irrelevant
        # Or it could be redefined to fit the anomaly detection context
        pass

    def findBestSplit(self, data: NDArray, targets: NDArray, *, weights: NDArray | None = None, classWeights: NDArray | None = None) -> tuple[int | None, float | None]:
        """
        selecting a random feature and threshold
        the most basic implementation
        """
        feature = np.random.randint(0, data.shape[1]) # feature selection

        minVal, maxVal = np.min(data[:, feature]), np.max(data[:, feature])
        threshold = np.random.uniform(minVal, maxVal) # threshold selection

        return feature, threshold
