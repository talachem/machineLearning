import numpy as np
from numpy.typing import NDArray


class RegressionScores:
    """
    This class handles performance scores for regression tasks
    """
    def __init__(self, numClasses: int | None = None, classNames: list[str] | None = None) -> None:
        self.name = self.__class__.__name__
        self.scores = ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'R-squared (R²)', 'Adjusted R-squared']
        self._scores = ['MSE', 'RMSE', 'MAE', 'RSQ']
        self.scoreLength = [len(item) for item in self._scores]

        # checking that numClasses and classNames are fitting to one another
        if numClasses is not None and classNames is not None:
            if len(classNames) != numClasses:
                raise ValueError('number of classes must be length of class names')

        # setting class names, if not provided
        if numClasses is not None or classNames is not None:
            self.numClasses = len(classNames) if numClasses is None else numClasses
            self.classNames = [f'Class {i}' for i in range(self.numClasses)] if classNames is None else classNames
        else:
            self.numClasses = 1
            self.classNames = [f'Class {i}' for i in range(self.numClasses)] if classNames is None else classNames

        self.nameLength = [len(item) for item in self.classNames]

    def MSE(self, predictions: NDArray, targets: NDArray, labels: NDArray | None = None) -> np.ndarray:
        """
        calculates the mean square error
        """
        if labels is None:
            # Calculate the square of differences
            squaredDiffs = (predictions - targets) ** 2

            # Calculate the mean of the squared differences
            mse = np.mean(squaredDiffs)
        else:
            mse = []
            for i in range(self.numClasses):
                index = np.where(labels == i)
                mse.append(self.MSE(predictions[index], targets[index]))
            mse = np.array(mse)

        return mse

    def RMSE(self, predictions: NDArray, targets: NDArray, labels: NDArray | None = None) -> np.ndarray:
        """
        calculates root mean square error
        """
        if labels is None:
            # Calculate MSE
            mse = self.MSE(predictions, targets)

            # Take the square root of the MSE
            rmse = np.sqrt(mse)
        else:
            rmse = []
            for i in range(self.numClasses):
                index = np.where(labels == i)
                rmse.append(self.RMSE(predictions[index], targets[index]))
            rmse = np.array(rmse)

        return rmse

    def MAE(self, predictions: NDArray, targets: NDArray, labels: NDArray | None = None) -> np.ndarray:
        """
        calculates mean absolute error
        """
        if labels is None:
            # Calculate the absolute differences
            absDiffs = np.abs(predictions - targets)

            # Calculate the mean of the absolute differences
            mae = np.mean(absDiffs)
        else:
            mae = []
            for i in range(self.numClasses):
                index = np.where(labels == i)
                mae.append(self.MAE(predictions[index], targets[index]))
            mae = np.array(mae)

        return mae

    def RSQ(self, predictions: NDArray, targets: NDArray, labels: NDArray | None = None) -> np.ndarray:
        """
        calculates r-squere error
        """
        if labels is None:
            # Compute residuals (actual minus predicted)
            residuals = targets - predictions

            # Compute the sum of squared residuals
            SSR = np.sum(residuals ** 2)

            # Compute the total sum of squares
            SST = np.sum((targets - np.mean(targets)) ** 2)

            # Compute R-squared (1 - the ratio of SSR to SST)
            R2 = 1 - (SSR / SST)
        else:
            R2 = []
            for i in range(self.numClasses):
                index = np.where(labels == i)
                R2.append(self.RSQ(predictions[index], targets[index]))
            R2 = np.array(R2)

        return R2

    def ARSQ(self, predictions: NDArray, targets: NDArray, numPredictors: int, labels: NDArray | None = None) -> np.ndarray:
        """
        calculates r-squere error
        """
        if labels is None:
            # Compute R-squared
            R2 = self.RSQ(predictions, targets)

            # Compute the number of data points
            numData = len(predictions)

            # Compute adjusted R-squared (accounting for the number of predictors)
            adjustedR2 = 1 - ((1 - R2) * (numData - 1) / (numData - numPredictors - 1))
        else:
            adjustedR2 = []
            for i in range(self.numClasses):
                index = np.where(labels == i)
                adjustedR2.append(self.ARSQ(predictions[index], targets[index], numPredictors))
            adjustedR2 = np.array(adjustedR2)

        return adjustedR2

    def calcScores(self, predictions: NDArray, targets: NDArray, labels: NDArray | None = None, numPredictors = None) -> None:
        """
        goes over the scores and calculates all of them
        """
        self.metrics = []

        # Calculate and store all metrics
        for metric in self._scores:
            metricValue = getattr(self, metric)(predictions, targets, labels)
            self.metrics.append(metricValue)

        # Calculate and store Adjusted R-squared only if numPredictors is provided
        self.numPredictors = numPredictors
        if self.numPredictors is not None:
            self.metrics.append(self.ARSQ(predictions, targets, numPredictors, labels))

    def __str__(self) -> str:
        # Create a string for the output
        printString = ""
        center = self.numClasses * (np.max(self.nameLength) + 2) + (np.max(self.scoreLength) + 1)
        if len(self.classNames) > 1:
            center += (np.max(self.nameLength) + 2)

        # Add the title for the metrics
        printString += ' Metrics '.center(center, '━') + '\n'
        printString += ' '.center(np.max(self.scoreLength) + 1)
        for name in self.classNames:
            printString += name.center(np.max(self.nameLength) + 2)
        if len(self.classNames) > 1:
            printString += 'total'.center(np.max(self.nameLength) + 2)
        printString += '\n'
        printString += '·' * (center) + '\n'

        for i, score in enumerate(self._scores):
            printString += score.rjust(np.max(self.scoreLength) + 1)
            if type(self.metrics[i]) is np.ndarray:
                for item in self.metrics[i]:
                    printString += str(np.round(item,2)).center(np.max(self.nameLength) + 2)
                printString += str(np.round(np.mean(self.metrics[i]),2)).center(np.max(self.nameLength) + 2)
            else:
                printString += str(np.round(self.metrics[i],2)).center(np.max(self.nameLength) + 2)
            printString += '\n'

        if self.numPredictors is not None:
            printString += 'ARSQ'.rjust(np.max(self.scoreLength) + 1)
            if type(self.metrics[-1]) is np.ndarray:
                for item in self.metrics[-1]:
                    printString += str(np.round(item,2)).center(np.max(self.nameLength) + 2)
            else:
                printString += str(np.round(self.metrics[-1],2)).center(np.max(self.nameLength) + 2)
        printString += '\n'

        return printString
