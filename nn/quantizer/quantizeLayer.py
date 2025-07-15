import numpy as np
from numpy._typing import NDArray
from numpy.typing import ArrayLike
from ..layer import Layer


class Quantize(Layer):
    """
    This class can be used to quantize data during a network forward call
    It can only be used in post training quantization
    It should be regarded as stop-gap, better pre quantize data
    """
    def __init__(self, bits: int = 8, scheme: str = "symmetric") -> None:
        super().__init__()
        self.bits = bits
        self.scheme = scheme
        # Determine qMin and qMax based on the quantization scheme and bit width
        if self.scheme == "symmetric":
            self.qMax = 2 ** (bits - 1) - 1
            self.qMin = -self.qMax
        elif self.scheme == "asymmetric":
            self.qMax = 2 ** bits - 1
            self.qMin = 0
        self.scale = 1
        self.zeroPoint = 0
        self.callibrated = False

    def forward(self, input: NDArray) -> NDArray:
        if not self.callibrated:
            if self.scheme == "asymmetric":
                # Adjust the scale and zeroPoint for asymmetric quantization
                data_min = np.min(input)
                data_max = np.max(input)

                # Scale calculation based on the actual range of the data
                self.scale = (data_max - data_min) / (self.qMax - self.qMin)

                # Zero-point calculation, ensuring it's within the quantized value range
                self.zeroPoint = np.round((self.qMin - data_min) / self.scale)
                self.zeroPoint = max(self.qMin, min(self.qMax, self.zeroPoint))
            else:
                # For symmetric quantization, scale is based on the maximum absolute value
                maxValue = np.max(np.abs(input))
                self.scale = (self.qMax - self.qMin) / (2 * maxValue)

            self.callibrated = True

        quantizedInput = np.round(input / self.scale + self.zeroPoint)
        clippedInput = np.clip(quantizedInput, self.qMin, self.qMax).astype(np.int32)

        return clippedInput

    def backward(self, gradient: NDArray) -> None:
        raise NotImplementedError("This layer can only be used for PTQ and does not support backpropagation.")

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        printString = self.name
        printString += '    bits: ' + str(self.bits)
        printString += '    scale: ' + str(self.scale)
        printString += '    zeroPoint: ' + str(self.zeroPoint)
        printString += '    scheme: ' + str(self.scheme)
        return printString


class Accumulator(Layer):
    """
    This is used for rescaling each layers out
    It can handle Asymmetric quantization
    """
    def __init__(self, bits: int, scheme: str, scale: int, zeroPoint: int) -> None:
        super().__init__()
        self.bits = bits
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.scheme = scheme

    def forward(self, input: NDArray) -> NDArray:
        input = np.round(input / self.scale) - self.zeroPoint
        return input.astype(np.int32)

    def backward(self, gradient: NDArray) -> None:
        raise NotImplementedError("This layer can only be used for PTQ and does not support backpropagation.")

    def __str__(self) -> str:
        """
        used for print the layer in a human readable manner
        """
        printString = self.name
        printString += '    bits: ' + str(self.bits)
        printString += '    scale: ' + str(self.scale)
        printString += '    zeroPoint: ' + str(self.zeroPoint)
        printString += '    scheme: ' + str(self.scheme)
        return printString
