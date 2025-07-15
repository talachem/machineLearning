import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class FeatureSelection(ABC):
    """
    Abstract base class for feature selection.
    """
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def __call__(self, features: NDArray, target: NDArray) -> NDArray:
        """
        A convenience method to run the feature selection.
        """
        return self._select(features, target)

    @abstractmethod
    def _select(self, features: NDArray, target: NDArray) -> NDArray:
        """
        This method needs to be implemented with different algorithms.
        """
        pass


class UsersChoice(FeatureSelection):
    """
    if the user already knows which feature to use, this will be useful
    it takes a list of feature indices
    """
    def __init__(self, featureIndices: list) -> None:
        super().__init__()
        self.featureIndices = featureIndices

    def _select(self, features: NDArray, target: NDArray) -> NDArray:
        """
        Selects features based on user-defined indices.
        """
        return self.featureIndices


class Variance(FeatureSelection):
    """
    Selects features with the largest variance.
    """
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def _select(self, features: NDArray, target: NDArray) -> NDArray:
        """
        Selects features with the largest variance.
        """
        variances = np.var(features, axis=0)
        return np.where(variances > self.threshold)[0]


class Random(FeatureSelection):
    """
    Selects features randomly.
    """
    def __init__(self, numberFeatures: int) -> None:
        super().__init__()
        self.numberFeatures = numberFeatures

    def _select(self, features: NDArray, target: NDArray) -> NDArray:
        """
        Selects features randomly.
        """
        nFeatures = features.shape[1]
        selected = np.random.choice(nFeatures, self.numberFeatures)
        return selected


class MutualInformation(FeatureSelection):
    """
    Selects features based on their mutual information with the target variable.
    """
    def __init__(self, numberFeatures: int) -> None:
        super().__init__()
        self.numberFeatures = numberFeatures

    def _select(self, features: NDArray, target: NDArray) -> NDArray:
        """
        Selects features based on their mutual information with the target variable.
        """
        mutualInformation = np.array([self._mutualInformation(features[:, i], target) for i in range(features.shape[1])])
        indices = np.argsort(mutualInformation)[::-1][:self.self.numberFeatures]
        return indices

    @staticmethod
    def _mutualInformation(feature: NDArray, targets: NDArray) -> float:
        """
        Computes the mutual information between two random variables..
        """
        # Compute the joint histogram of x and y
        joinedHistogram = np.histogram2d(feature, targets, bins=(len(np.unique(feature)), len(np.unique(targets))))[0]

        # Compute the histogram of x and y
        featureHistogram = np.histogram(feature, bins=len(np.unique(feature)))[0]
        targetsHistogram = np.histogram(targets, bins=len(np.unique(targets)))[0]

        # Normalize the histograms
        joinedHistogramNormalized = joinedHistogram / len(feature)
        featureHistogramNormalized = featureHistogram / len(feature)
        targetHistogramNormalized = targetsHistogram / len(feature)

        # Compute the logarithmic term for mutual information
        logarithmTerm = np.zeros_like(joinedHistogram)
        logarithmTerm[joinedHistogram != 0] = joinedHistogramNormalized[joinedHistogramNormalized != 0] / (featureHistogramNormalized[np.newaxis, :] * targetHistogramNormalized[:, np.newaxis])[joinedHistogramNormalized != 0]

        # Compute the mutual information
        mutualInfo = (joinedHistogram * np.log2(logarithmTerm)).sum()

        return mutualInfo


class ANOVA(FeatureSelection):
    """
    A class for selecting features based on the analysis of variance (ANOVA) test.
    """
    def __init__(self, numberFeatures: int) -> None:
        super().__init__()
        self.numberFeatures = numberFeatures

    def _select(self, features: NDArray, target: NDArray) -> NDArray:
        """
        Selects the `numberFeatures` features with the highest F-values based on the ANOVA test.
        """
        args = [features[:, i][target == j] for j in np.unique(target) for i in range(features.shape[1])]
        fValues, pValues = self._fOneway(*args)
        indices = np.argsort(fValues)[::-1][:self.numberFeatures]
        return indices

    @staticmethod
    def fOneWay(*args: NDArray) -> tuple[float, NDArray]:
        """
        Performs a one-way ANOVA test.
        """
        # Get the number of groups and total number of samples
        numGroups = len(args)
        numSamples = np.sum([len(x) for x in args])

        # Compute the mean across all groups
        meanTotal = np.mean(np.concatenate(args))

        # Compute the sum of squares between groups and within groups
        ssBetween = np.sum([len(x) * (np.mean(x) - meanTotal)**2 for x in args])
        ssWithin = np.sum([np.sum((x - np.mean(x))**2) for x in args])

        # Compute degrees of freedom for between groups and within groups
        dfBetween = numGroups - 1
        dfWithin = numSamples - numGroups

        # Compute mean square values for between groups and within groups
        msBetween = ssBetween / dfBetween
        msWithin = ssWithin / dfWithin

        # Compute the F-value
        fValue = msBetween / msWithin

        # Generate a distribution of F-values and compute the p-value
        pValue = 1 - np.array([np.sum(np.random.f(dfBetween, dfWithin, size=100000) <= fValue) / 100000 for _ in range(1000)])

        return fValue, pValue


class KendallTau(FeatureSelection):
    """
    A class for selecting features based on their Kendall's tau correlation with the target variable.
    """
    def __init__(self, targetFeature: int = None, threshold: float = 0.) -> None:
        super().__init__()
        self.targetFeature = targetFeature
        self.threshold = threshold

    def _select(self, features: NDArray, target: NDArray) -> NDArray:
        """
        Selects the features with Kendall's tau correlation greater than `threshold` with the target variable.
        """
        numFeatures = features.shape[1]

        # Choose a random target feature if none is specified
        targetFeature = self.targetFeature if self.targetFeature is not None else np.random.randint(0, numFeatures)

        # Calculate Kendall's tau scores for all features
        scores = np.zeros(numFeatures)
        for i in range(numFeatures):
            scores[i] = abs(self._kendallTau(features[:, i], features[:, targetFeature]))

        return np.where(scores > self.threshold)[0]

    @staticmethod
    def _kendallTau(feature: NDArray, compare: NDArray) -> float:
        """
        Computes the Kendall's tau correlation between two random variables.
        """
        n = len(feature)

        # Ensure that the feature and compare arrays have the same length
        assert n == len(compare), "feature and compare arrays must have the same length"

        # Initialize counters for number of concordant, discordant pairs and ties for both arrays
        concordant = 0
        discordant = 0
        tiesFeatures = 0
        tiesCompare = 0

        # Loop over all pairs of elements in the arrays
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Check if elements in the same position of both arrays are equal
                if feature[i] == feature[j]:
                    if compare[i] == compare[j]:
                        tiesFeatures += 1
                        tiesCompare += 1
                    elif compare[i] < compare[j]:
                        concordant += 0.5
                        tiesFeatures += 1
                    else:
                        discordant += 0.5
                        tiesFeatures += 1
                    continue
                # Check if elements in the same position of the two arrays are different and
                # determine if they are concordant or discordant pairs
                if compare[i] == compare[j]:
                    if feature[i] < feature[j]:
                        concordant += 0.5
                        tiesCompare += 1
                    else:
                        discordant += 0.5
                        tiesCompare += 1
                    continue
                if feature[i] < feature[j] and compare[i] < compare[j]:
                    concordant += 1
                elif feature[i] > feature[j] and compare[i] > compare[j]:
                    concordant += 1
                else:
                    discordant += 1

        # Compute Kendall's tau correlation coefficient and return it
        tau = (concordant - discordant) / np.sqrt(
            (concordant + discordant + tiesFeatures) * (concordant + discordant + tiesCompare)
        )
        return tau
