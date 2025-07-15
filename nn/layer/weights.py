from numpy.typing import NDArray
import numpy as np


def initializeWeights(size: tuple | int, scale: float = 1.0, init: str = 'random') -> NDArray:
    """
    Initialize filter using a normal distribution with and a
    standard deviation inversely proportional the square root of the number of units
    """
    if init == 'random':
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc=0, scale=stddev, size=size)
    elif init == 'ones':
        return np.ones(size)
    elif init == 'zeros':
        return np.zeros(size)
    else:
        raise ValueError('not a valid init argument')


class Weights:
    """
    the idea behind class is to combine everything an optimizer needs into one object
    this way layers and optimizers don't need to take care of storing and providing
    things like previous updates or cache
    """
    __slots__ = ['_values', '_quantizedValues', 'prevValues', 'deltas', 'prevDeltas', 'cache', 'scale', 'maxValue', '_useQuantization', 'zeroPoint', 'bits', 'qMin', 'qMax']

    def __init__(self, size: tuple | int, values: NDArray | None = None, init: str = 'random') -> None:
        self._values = initializeWeights(size, init=init) if values is None else values
        self.prevValues = None
        self.deltas = np.zeros(size)
        self.prevDeltas = None
        self.cache = None

        self._quantizedValues = np.zeros(size)
        self.scale = 1
        self.maxValue = 0
        self.zeroPoint = 0
        self._useQuantization = False

    @property
    def values(self):
        """
        Depending on the _useQuantized flag, return either the original
        or quantized (and dequantized back) weight values for computation.
        """
        if self._useQuantization:
            return self._quantizedValues
        else:
            return self._values

    @values.setter
    def values(self, newValues):
        """
        Allow updates to the weight values directly.
        """
        self._values = newValues

    @property
    def qualifiedName(self) -> tuple:
        return self.__class__.__module__, self.__class__.__name__

    def toDict(self) -> dict:
        saveDict = {}
        saveDict['size'] = self._values.shape
        saveDict['values'] = self._values.tolist()
        saveDict['deltas'] = self.deltas.tolist()
        if self.prevValues is not None:
            saveDict['prevValues'] = self.prevValues.tolist()
        saveDict['cache'] = {}
        if type(self.cache) == dict:
            saveDict['cache']['values'] = {}
            for key in self.cache:
                saveDict['cache']['values'][key] = self.cache[key].tolist()
            saveDict['cache']['type'] = 'dict'
        elif type(self.cache) == np.ndarray:
            saveDict['cache']['values'] = self.cache.tolist()
            saveDict['cache']['type'] = 'np.ndarray'

        if self._useQuantization:
            saveDict['quantization'] = {}
            saveDict['quantization']['scale'] = int(self.scale)
            saveDict['quantization']['qMin'] = int(self.qMin)
            saveDict['quantization']['qMax'] = int(self.qMax)
            saveDict['quantization']['zeroPoint'] = int(self.zeroPoint)
            saveDict['quantization']['maxValue'] = float(self.maxValue)
            saveDict['quantization']['quantizedValues'] = self._quantizedValues.tolist()

        return saveDict

    def fromDict(self, loadDict: dict) -> None:
        self._values = np.array(loadDict['values'])
        self.deltas = np.array(loadDict['deltas'])
        if 'prevValues' in loadDict:
            self.prevValues = np.array(loadDict['prevValues'])
        if loadDict['cache']['type'] == 'np.ndarray':
            self.cache = np.array(loadDict['cache']['values'])
        elif loadDict['cache']['type'] == 'dict':
            self.cache = {}
            for key in loadDict['cache']['values']:
                self.cache[key] = np.array(loadDict['cache']['values'][key])

        if 'quantization' in loadDict:
            self.scale = loadDict['quantization']['scale']
            self.maxValue = loadDict['quantization']['maxValue']
            self.zeroPoint = loadDict['quantization']['zeroPoint']
            self.qMin = loadDict['quantization']['qMin']
            self.qMax = loadDict['quantization']['qMax']
            self._quantizedValues = np.array(loadDict['quantization']['quantizedValues'])
            self._useQuantization = True
        else:
            self._useQuantization = False

    def quantize(self, bits: int = 8, scheme: str = "symmetric") -> None:
        """
        Quantizes the weight values to a specified bit width.
        """
        self.zeroPoint = 0  # Zero-point is used in asymmetric quantization
        self.bits = bits

        # Determine qMin and qMax based on the quantization scheme and bit width
        if scheme == "symmetric":
            self.qMax = 2 ** (bits - 1) - 1
            self.qMin = - 2 ** (bits - 1)
        elif scheme == "asymmetric":
            self.qMax = 2 ** bits - 1
            self.qMin = 0
        else:
            raise ValueError(f"{scheme} is not a recognized quantization scheme")

        if scheme == "asymmetric":
            # Adjust the scale and zeroPoint for asymmetric quantization
            data_min = np.min(self._values)
            data_max = np.max(self._values)

            # Scale calculation based on the actual range of the data
            self.scale = (data_max - data_min) / (self.qMax - self.qMin)

            # Zero-point calculation, ensuring it's within the quantized value range
            self.zeroPoint = np.round((self.qMin - data_min) / self.scale)
            self.zeroPoint = max(self.qMin, min(self.qMax, self.zeroPoint))
        else:
            # For symmetric quantization, scale is based on the maximum absolute value
            self.maxValue = np.max(np.abs(self._values))
            self.scale = (self.qMax - self.qMin) / (2 * self.maxValue)

        # Apply quantization
        if scheme == "asymmetric":
            self._quantizedValues = np.round(self._values * self.scale) + self.zeroPoint
        else:
            self._quantizedValues = np.round(self._values * self.scale)

        self._quantizedValues = np.clip(self._quantizedValues, self.qMin, self.qMax).astype(np.int32)  # Ensure values are within range

        self._useQuantization = True

    def dequantize(self) -> NDArray:
        """
        Dequantizes the weight values back to floating point.
        """
        if self._useQuantization:
            # For both symmetric and asymmetric, the formula below applies because
            # for symmetric quantization, zeroPoint is 0.
            # Note: Ensure self.scale and self.zeroPoint are correctly set during quantization.
            return (self._quantizedValues - self.zeroPoint) * self.scale
        else:
            # If not quantized, simply return the original values.
            return self._values

    @property
    def quantizationRange(self) -> tuple[int, int]:
        return self.qMin, self.qMax

    def __str__(self) -> str:
        printString = ""
        printString += str(self.values)
        return printString
