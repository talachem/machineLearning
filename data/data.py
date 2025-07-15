import numpy as np
from .dataLoader import DataSet, DataLoader
from itertools import product
from numpy.typing import NDArray


def concatenate(dictionary: dict) -> NDArray:
    """
    This function converts a dictionary of arrays into a single array.
    It works properly only if all entries in the dict have the same shape.
    """
    return np.concatenate([dictionary[name] for name in dictionary.keys()])


def toCategorical(targets: NDArray) -> NDArray:
    """
    Converts a vector of labels into one-hot encoding format
    """
    length = len(targets)
    pos = np.expand_dims(targets, axis=1).astype(int)
    numberOfLabel = np.max(pos) + 1

    categorical = np.zeros(shape=(length, numberOfLabel), dtype=int)
    np.put_along_axis(categorical, indices=pos, values=1, axis=1)

    return categorical


def genData(amount: int, number: int = 1, direction: str = 'vertical') -> NDArray:
        """
        generates fake data for testing
       """
        data = []
        choices = np.arange(0,9)
        for i in range(amount):
            index = np.random.choice(choices, number, replace=False)
            img = np.random.rand(9,9)
            if direction == 'vertical':
                img[:,index] += np.random.randint(2,4,(9,number)) + np.random.rand(9,number) - 0.5
            elif direction == 'horizontal':
                img[index] += np.random.randint(2,4,(number,9)) + np.random.rand(number,9) - 0.5
            elif direction == 'both':
                img[:,index] += np.random.randint(2,4,(9,number)) + np.random.rand(9,number) - 0.5
                img[index] += np.random.randint(2,4,(number,9)) + np.random.rand(number,9) - 0.5
            data.append(img)
        return np.array(data)


def genCoordinates(amount: int, radius: float = 10., xcenter: float = 0., ycenter: float = 0., noiseScale: float = 0.02) -> NDArray:
    """
    generate coordinates for fake data
    """
    points = []
    for i in range(amount):
        # radius of the circle
        r = radius * np.random.uniform(1 - noiseScale, 1 + noiseScale)

        # random angle
        alpha = 2 * np.pi * np.random.uniform(0, 1)

        # calculating coordinates
        x = r * np.cos(alpha) + xcenter
        y = r * np.sin(alpha) + ycenter

        points.append([x,y])

    return np.array(points)


def flatten(List: list[list]) -> list:
    """
    this function flattens python lists, since they can be ragged and numpy/pytorch have troubles handling them
    this is used in two places in the code, here and in settings
    """
    returnList = []
    for item in List:
        if type(item) == list:
            elements = flatten(item)
            for element in elements:
                returnList.append(element)
        else:
            returnList.append(item)
    return returnList


class Data:
    """
    the purpose of this class is to handle loading data from the disk,
    splitting it into an evaluation and training data set
    this class is much more specifically tailor made for purposes of
    handeling belle 2 pxd data
    """
    def __init__(self, trainAmount: int, evalAmount: int, batchSize: int = 32, kFold: int = 1, dataPath: str = '.', normalize: bool = False) -> None:
        self.trainAmount = trainAmount
        self.evalAmount = evalAmount
        self.batchSize = batchSize
        self.kFold = kFold
        self.dataPath = dataPath
        self.normalize = normalize
        self._names = []
        self._nameLength = []
        self.features = ['cluster', 'xPosition', 'yPosition', 'zPosition']

        # these are all available features of pxd data sets (some are exclusive to MC)
        self._allFeatures = {'event': ['event'],
                         'clsNumber': ['clsNumber'],
                           'cluster': ['cluster'],
                     'clsParameters': ['clsCharge', 'seedCharge', 'clsSize', 'uSize', 'vSize'],
                       'coordinates': ['xPosition', 'yPosition', 'zPosition'],
                        'uvPosition': ['uPosition', 'vPosition'],
                               'pdg': ['pdg']}
        self._keyLength = []
        self._featuresLength = []
        for key in self._allFeatures:
            self._keyLength.append(len(key))
            for item in self._allFeatures[key]:
                self._featuresLength.append(len(item))

    def inputFeatures(self, *features: str) -> None:
        """
        for selecting which features should be kept after importing
        """
        self.features = [self._allFeatures[feature] for feature in features]
        self.features = flatten(self.features)

    def _importData(self, fileName: str) -> tuple[NDArray, NDArray]:
        """
        this handels this import of a single data file and splitting it into en eval/train set
        don't call this function outside of the class, 'importData' class this for importing
        individual files
        """
        imported = np.load(f'{self.dataPath}/{fileName}.npy', allow_pickle=True)
        if self.trainAmount + self.evalAmount > len(imported):
            raise ValueError(f'the amount of requested data exceeds the avaible amount for file {fileName}')

        # converting the structured array into simple array
        # also selecting features from the array
        imported = np.concatenate([imported[feature].reshape(len(imported),-1) for feature in self.features], axis=1)
        indices = np.random.permutation(len(imported))
        imported = imported[indices]

        return imported[0:self.trainAmount], imported[self.trainAmount:self.trainAmount+self.evalAmount]

    def importData(self, *fileNames: str) -> None:
        """
        this handels all the imports and converting into dataloaders
        the dataloaders will then serve as iter objects
        this method uses '_importData' to import individual files
        """
        trainData, evalData, trainLabels, evalLabels = {}, {}, {}, {}

        # importing files
        for i, name in enumerate(fileNames):
            self._names.append(name)
            self._nameLength.append(len(name))
            trainData[name], evalData[name] = self._importData(name)

            # creating one-hot labels
            trainLabels[name] = np.full(self.trainAmount, i, dtype=int) # creating labels for training
            evalLabels[name] = np.full(self.evalAmount, i, dtype=int) # creating labels for evaluation

        # concatenating and shuffling training data
        indices = np.random.permutation(len(fileNames) * self.trainAmount)
        trainData = concatenate(trainData)
        trainLabels = concatenate(trainLabels)
        trainData = trainData[indices]
        trainLabels = toCategorical(trainLabels[indices]) # converting labels to one-hot encoding

        # concatenating evaluation data
        evalData = concatenate(evalData)
        if self.normalize is True:
            minValues = np.min((trainData.min(0), evalData.min(0)), axis=0)
            maxValues = np.max((trainData.max(0), evalData.max(0)), axis=0)
            trainData = (trainData - minValues) / (maxValues - minValues)
            evalData = (evalData - minValues) / (maxValues - minValues)
        evalLabels = toCategorical(concatenate(evalLabels)) # converting labels to one-hot encoding

        # converting np arrays to DataSets
        trainSet = DataSet(trainData, labels=trainLabels)
        evalSet = DataSet(evalData, labels=evalLabels)

        # creating DataLoaders of DataSets
        if self.batchSize is not None:
            self._train = DataLoader(trainSet, self.batchSize, shuffle=True, kFold=self.kFold)
            self._eval = DataLoader(evalSet, 2 * self.batchSize)
        else:
            self._train = DataLoader(trainSet, 1, shuffle=True, kFold=self.kFold)
            self._eval = DataLoader(evalSet, 1)

    def generateTestData(self, sets: int = 2) -> None:
        """
        this creates dummy data for testing networks
        the data is of similar shape as pxd data, but
        very easily separable
        """
        datasets = list(product(range(1,sets), ['vertical', 'horizontal']))[:sets]
        trainData, evalData, trainLabels, evalLabels = {}, {}, {}, {}
        for i, item in enumerate(datasets):
            self._names.append(f'class {i}')
            self._nameLength.append(len(f'class {i}'))
            trainData[i] = genData(self.trainAmount, item[0], item[1])
            evalData[i] = genData(self.evalAmount, item[0], item[1])
            trainLabels[i] = np.full(self.trainAmount, i)
            evalLabels[i] = np.full(self.evalAmount, i)

        indices = np.random.permutation(sets * self.trainAmount)
        trainData = concatenate(trainData)
        trainLabels = concatenate(trainLabels)
        trainData = trainData[indices]
        trainLabels = toCategorical(trainLabels[indices])

        evalData = concatenate(evalData)
        evalLabels = toCategorical(concatenate(evalLabels))

        trainSet = DataSet(trainData, labels=trainLabels)
        evalSet = DataSet(evalData, labels=evalLabels)

        self._train = DataLoader(trainSet, self.batchSize, shuffle=True, kFold=self.kFold)
        self._eval = DataLoader(evalSet, 2 * self.batchSize)

    def fold(self) -> None:
        """
        k-fold the training data set/loader
        """
        self._train.fold()

    def trainMode(self) -> None:
        """
        setting the training set/loader to train mode
        """
        self._train.train()

    def evalMode(self) -> None:
        """
        setting the training set/loader to eval mode
        """
        self._train.eval()

    @property
    def trainSet(self) -> DataSet:
        return self._train.dataSet

    @property
    def evalSet(self) -> DataSet:
        return self._eval.dataSet

    @property
    def train(self) -> DataLoader:
        return self._train

    @property
    def eval(self) -> DataLoader:
        return self._eval

    def __str__(self) -> str:
        """
        this function gets called when this class is printed, it will print/write some numerical values about the data set
        it reads the actual metrics from the dataset, instead of reprinting the config from settings
        """
        center = 75
        align = 15
        printString = ' Data '.center(center, '━') + '\n'
        #printString += ' Sizes '.center(center, '—') + '\n'
        printString += ' '.ljust(max(self._nameLength)) + 'training'.center(align) + 'validation'.center(align) + 'total'.center(align) + '\n'

        printString += '·'*center + '\n'
        for i, label in enumerate(self._train.dataSet.uniques):
            name = self._names[i]
            train = int(len(np.where(self._train.dataSet.labels == label)[0]) / len(label))
            valid = int(len(np.where(self._eval.dataSet.labels == label)[0]) / len(label))
            total = train + valid
            printString += name.rjust(max(self._nameLength)) + str(train).center(align) + str(valid).center(align) + str(total).center(align) + '\n'

        printString += '·'*center + '\n'
        train = len(self._train.dataSet.labels)
        valid = len(self._eval.dataSet.labels)
        total = train + valid
        printString += 'sums'.rjust(max(self._nameLength)) + str(train).center(align) + str(valid).center(align) + str(total).center(align) + '\n'

        #printString += ' k-Folds '.center(center, '—') + '\n'
        if len(self._train._order) > 1:
            printString += '·' * center + '\n'
            for i in self._train._order:
                printString += f'{i}. fold'.rjust(max(self._nameLength)) + str(self._train._lengthes[i]).center(align)
                if i == len(self._eval._lengthes) - 1:
                    printString += str(self._eval._lengthes[i]).center(align)
                printString += '\n'

        #printString += ' batches '.center(center, '—') + '\n'
        if self.batchSize is not None:
            printString += '·'*center + '\n'
            printString += 'batch size'.rjust(max(self._nameLength)) + str(self._train.batchSize).rjust(5).center(align) + str(self._eval.batchSize).rjust(5).center(align) + '\n'
            printString += 'batches'.rjust(max(self._nameLength)) + str(len(self._train)).rjust(5).center(align) + str(len(self._eval)).rjust(5).center(align) + '\n'
            trainLastBatch = len(self._train.dataSet.labels) % self._train.batchSize
            validLastBatch = len(self._eval.dataSet.labels) % self._eval.batchSize
            printString += 'last batch'.rjust(max(self._nameLength)) + str(trainLastBatch).rjust(5).center(align) + str(validLastBatch).rjust(5).center(align) + '\n'
        if len(self._train._order) > 1:
            printString += 'k-fold'.rjust(max(self._nameLength)) + str(self._train.kFold).rjust(5).center(align) + str(self._eval.kFold).rjust(5).center(align) + '\n'

        return printString + '\n'
