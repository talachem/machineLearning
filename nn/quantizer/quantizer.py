import numpy as np
from collections import namedtuple
from copy import deepcopy
from ..module import Module
from ..layer import Layer
from .quantizeLayer import Quantize, Accumulator


# Define the named tuple type outside your class
QuantizationError = namedtuple('QuantizationError', ['roundingError', 'clippingError'])


class Quantizer:
    """
    A class that can take a network/module and quantize it post training.
    """
    def __init__(self, bits: int = 8, *, perChannel: bool = False, scheme: str = "symmetric", overWriteActiScheme: bool = False) -> None:
        """
        Initializes the quantizer with a network/module to quantize.

        Parameters:
            module (Module): The network/module to be quantized.
            bits (int): The bit width for quantization.
        """
        self.bits = bits
        self.perChannel = perChannel
        self.scheme = scheme
        self.overWrite = overWriteActiScheme

    def callibrate(self) -> None:
        """
        this callibrates and minimizes (pareto) quantization errors
        """
        pass

    def addQuantizeLayer(self, module: Module) -> None:
        """
        adds a layer that automatically quantizes into the data flow of the module
        """
        module.insert(0, Quantize(bits=self.bits, scheme=self.scheme))

    @property
    def quantizationError(self, quantizedModule: Module) -> QuantizationError:
        """
        this returns the two main errors of the quantization
        """
        return QuantizationError(self._roundingError(quantizedModule), self._clippingError(quantizedModule))

    def _roundingError(self, quantizedModule: Module) -> float:
        """
        A private method for calculating the mean absolute rounding error.
        """
        totalError = 0.
        totalElements = 0

        for layer in quantizedModule:
            try:
                params = layer.params()
            except AttributeError:
                # 'params' method not found in the layer, skip updating
                continue

            for param in params:
                dequantizedWeights = param.dequantize()
                errorWeights = np.abs(param._values - dequantizedWeights)
                totalError += np.sum(errorWeights)
                totalElements += np.prod(param._values.shape)

        # Calculate the mean absolute error
        meanError = totalError / totalElements if totalElements > 0 else 0
        return meanError

    def _clippingError(self, quantizedModule: Module) -> float:
        totalClippingError = 0.
        totalElements = 0

        for layer in quantizedModule:
            try:
                params = layer.params()
            except AttributeError:
                # 'params' method not found in the layer, skip updating
                continue

            for param in layer.params():
                # Assuming you have a method or a way to determine Q_min and Q_max for each parameter
                qMin, qMax = param.quantizationRange

                # Calculate clipping error for values below qMin
                belowMin = np.minimum(param._values - qMin, 0)
                # Calculate clipping error for values above qMax
                aboveMax = np.maximum(param._values - qMax, 0)

                # Sum of absolute errors gives total clipping error for the parameter
                paramClippingError = np.sum(np.abs(belowMin) + np.abs(aboveMax))
                totalClippingError += paramClippingError

                # Update total elements for averaging
                totalElements += np.prod(param._values.shape)

        # Compute mean clipping error if needed, or return total
        meanClippingError = totalClippingError / totalElements if totalElements > 0 else 0
        return meanClippingError

    def __call__(self, module: Module) -> Module:
        """
        Applies quantization to all quantizable parameters in the module.
        """
        qunaitzedModule = deepcopy(module)
        lastLayer = len(module)
        for i, layer in enumerate(qunaitzedModule):
            self._quantizeLayer(layer)
            if hasattr(layer, 'weights') and i < lastLayer:
                bits = layer.weights.bits
                scale = layer.weights.scale
                zeroPoint = layer.weights.zeroPoint
                qunaitzedModule.insert(i+1, Accumulator(bits, self.scheme, scale, zeroPoint))

        return qunaitzedModule

    def dequantize(self, quatizedModule: Module) -> Module:
        """
        Applies dequantization to all dequantizable parameters in the module.
        """
        for layer in quatizedModule:
            self._dequantizeLayer(layer)

        return quatizedModule

    def _quantizeLayer(self, layer: Layer) -> None:
        """
        Quantizes the weights (and biases) of a single layer if applicable,
        or the layer itself if it supports direct quantization.
        """
        if hasattr(layer, 'params'):
            # For layers with parameters like weights and biases
            params = layer.params()
            for param in params:
                param.quantize(bits=self.bits, scheme=self.scheme)
        elif hasattr(layer, 'quantize'):
            # For layers supporting direct quantization, like activation layers with LUT
            layer.quantize(bits=self.bits, scheme=self.scheme, overWriteScheme=self.overWrite)

    def _dequantizeLayer(self, layer: Layer) -> None:
        """
        Dequantizes the weights (and biases) of a single layer if applicable,
        or the layer itself if it supports direct dequantization.
        """
        if hasattr(layer, 'params'):
            # For layers with parameters like weights and biases
            params = layer.params()
            for param in params:
                param.dequantize()
        elif hasattr(layer, 'dequantize'):
            # For layers supporting direct dequantization, if applicable
            layer.dequantize()
