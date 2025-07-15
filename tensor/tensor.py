import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Callable
from abc import ABC, abstractmethod
from functools import partial
#from .backend import BackendInterface, NumpyBackend, CupyBackend, NumbaBackend


class Tensor():
    __slots__ = ['_backend', 'data', 'gradient', 'requireGradient', 'gradientFunc', 'batched']

    #__backend__ = NumpyBackend()

    def __init__(self, data: Any,
                 gradient: Any = None,
                 gradientFunc: Callable = None,
                 requireGradient: bool = False,
                 batched: bool = True) -> None:

        #self._backend = Tensor.__backend__

        if isinstance(data, (list | np.ndarray)):
            data = np.array(data)
        elif isinstance(data, (int, float)):
            data = np.array([data])
        elif isinstance(data, self.__class__):
            gradient = data.gradient if gradient is None else gradient
            gradientFunc = data.gradientFunc if gradientFunc is None else gradientFunc
            requireGradient = data.requireGradient if requireGradient is False else requireGradient
            data = data.data

        if len(data.shape) == 1:
            data = np.reshape(data, (1, *data.shape))

        self.data = data
        self.gradient = gradient
        self.requireGradient = requireGradient
        self.gradientFunc = gradientFunc
        self.batched = batched

    def zeroGradient(self) -> None:
        """In-place operation for nulling the gradient"""
        if self.requireGradient:
            self.gradient = np.zeros_like(self.data)
        else:
            raise AttributeError("this tensor is not differentiable")

    def backward(self, gradient=None):
        """
        Compute the gradients recursively by applying the chain rule.
        """
        # If grad_fn is not set, this is probably the starting point for backpropagation,
        # so we don't need to compute further backward.
        if self.gradientFunc is None:
            return

        if gradient is None:
            gradient = np.ones_like(self.data)

        if self.gradient:
            # Accumulate gradients instead of overwriting.
            self.gradient += gradient
        else:
            self.gradient = gradient

        # Compute the local gradients using grad_fn
        self.gradientFunc(gradient)

    def __repr__(self) -> str:
        """String representation."""
        dataTitle = 'data:\n'
        gradientTitle = 'gradient:\n'
        dataStr = str(self.data)
        gradientStr = str(self.gradient)
        if self.requireGradient is True:
            return dataTitle + dataStr + '\n' + gradientTitle + gradientStr
        else:
            return dataTitle + dataStr

    def copy(self) -> 'Tensor':
        data = np.copy(self.data)
        gradient = np.copy(self.gradient)
        return self.__class__(data, gradient, gradientFunc=self.gradientFunc, requireGradient=self.requireGradient)

    @property
    def strides(self) -> tuple:
        return self.data.strides

    def __len__(self) -> int:
        """Return the length of the value."""
        return len(self.data)

    @property
    def shape(self) -> tuple:
        """Return the shape of the value."""
        return self.data.shape

    @property
    def ndim(self) -> tuple:
        """Return the ndim of the value."""
        return self.data.ndim

    def reshape(self, newshape) -> 'Tensor':
        return reshapeForward(self, newshape)

    def transpose(self) -> 'Tensor':
        return transposeForward(self)

    def T(self) -> 'Tensor':
        return transposeForward(self)

    def tolist(self) -> tuple[list, list] | list:
        if self.requireGradient is True:
            return self.data.tolist(), self.gradient.tolist()
        else:
            return self.data.tolist()

    #@classmethod
    #def setBackend(cls, backend: BackendInterface) -> None:
    #    if isinstance(backend, BackendInterface):
    #        cls.__backend__ = backend
    #    else:
    #        raise TypeError(f"{backend} is not an backend")

    def __getitem__(self, index):
        """Get an item by index."""
        if self.requireGradient is True and self.gradient:
            return self.__class__(data=self.data[index], gradient=self.gradient[index], requireGradient=True, gradientFunc=self.gradientFunc)
        elif self.requireGradient is True:
            return self.__class__(data=self.data[index], requireGradient=True, gradientFunc=self.gradientFunc)
        else:
            return self.__class__(data=self.data[index], requireGradient=False)

    def __setitem__(self, index, value) -> None:
        """Set the value of an item by index."""
        if isinstance(value, self.__class__):
            self.data[index] = value.data
            if self.requireGradient is True and self.gradient:
                self.gradient[index] = value.gradient
                self.requireGradient = True
        else:
            self.data[index] = value
            self.gradient[index] = 0

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            operation = ufuncMap.get(ufunc)
            if operation is not None:
                return operation(*inputs, **kwargs)
        raise NotImplementedError(f'{ufunc} is not implemented yet')

    def __array_function__(self, func, types, args, kwargs):
        operation = funcMap.get(func)
        if operation is not None:
            return operation(*args, **kwargs)
        raise NotImplementedError(f'{func} is not implemented yet')

    def __add__(self, other: ArrayLike) -> 'Tensor':
        return addForward(self, other)

    def __radd__(self, other: ArrayLike) -> 'Tensor':
        return addForward(other, self)

    def __iadd__(self, other: ArrayLike) -> 'Tensor':
        result = addForward(self, other)
        self.data = result.data
        self.gradient = result.gradient
        self.requireGradient = result.requireGradient
        return self

    def __sub__(self, other: ArrayLike) -> 'Tensor':
        return subtractForward(self, other)

    def __rsub__(self, other: ArrayLike) -> 'Tensor':
        return subtractForward(other, self)

    def __isub__(self, other: ArrayLike) -> 'Tensor':
        result = subtractForward(self, other)
        self.data = result.data
        self.gradient = result.gradient
        self.requireGradient = result.requireGradient
        return self

    def __mul__(self, other: ArrayLike) -> 'Tensor':
        return multiplyForward(self, other)

    def __rmul__(self, other: ArrayLike) -> 'Tensor':
        return multiplyForward(other, self)

    def __imul__(self, other: ArrayLike) -> 'Tensor':
        result = multiplyForward(self, other)
        self.data = result.data
        self.gradient = result.gradient
        self.requireGradient = result.requireGradient
        return self

    def __truediv__(self, other: ArrayLike) -> 'Tensor':
        return divideForward(self, other)

    def __rtruediv__(self, other: ArrayLike) -> 'Tensor':
        return divideForward(other, self)

    def __itruediv__(self, other: ArrayLike) -> 'Tensor':
        result = divideForward(self, other)
        self.data = result.data
        self.gradient = result.gradient
        self.requireGradient = result.requireGradient
        return self

    def __matmul__(self, other: ArrayLike) -> 'Tensor':
        return matmulForward(self, other)

    def __rmatmul__(self, other: ArrayLike) -> 'Tensor':
        return matmulForward(other, self)

    def __imatmul__(self, other: ArrayLike) -> 'Tensor':
        result = matmulForward(self, other)
        self.data = result.data
        self.gradient = result.gradient
        self.requireGradient = result.requireGradient
        return self

    def __pow__(self, other: ArrayLike) -> 'Tensor':
        return powerForward(self, other)

    def __rpow__(self, other: ArrayLike) -> 'Tensor':
        return powerForward(other, self)

    def __ipow__(self, other: ArrayLike) -> 'Tensor':
        result = powerForward(self, other)
        self.data = result.data
        self.gradient = result.gradient
        self.requireGradient = result.requireGradient
        return self

    def __abs__(self) -> 'Tensor':
        return absForward(self)

    def __pos__(self) -> 'Tensor':
        return positiveForward(self)

    def __neg__(self) -> 'Tensor':
        return negativeForward(self)

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        return equalForward(self, other)

    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        return greaterForward(self, other)

    def __ge__(self, other) -> bool:
        """Greater than or equal to comparison."""
        return greaterEqualForward(self, other)

    def __lt__(self, other) -> bool:
        """Less than comparison."""
        return lessForward(self, other)

    def __le__(self, other) -> bool:
        """Less than or equal to comparison."""
        return lessEqualForward(self, other)

    def sum(self, axis=None, dtype=None, keepdims=False) -> 'Tensor':
        return sumForward(self, axis, dtype, keepdims)

    def prod(self, axis=None, dtype=None, keepdims=False) -> 'Tensor':
        return prodForward(self, axis, dtype, keepdims)

    def max(self, axis=None, keepdims=False) -> 'Tensor':
        return maxForward(self, axis, keepdims)

    def min(self, axis=None, keepdims=False) -> 'Tensor':
        return minForward(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        return meanForward(self, axis, keepdims)

    def var(self, axis=None, ddof=0, keepdims=False) -> 'Tensor':
        return varForward(self, axis, ddof, keepdims)

    def std(self, axis=None, keepdims=False) -> 'Tensor':
        return stdForward(self, axis, keepdims)


def checkTensor(tensor: Tensor) -> Tensor:
    if isinstance(tensor, Tensor):
        return tensor
    return Tensor(tensor)


#
# Two Tensors
#


#def getbroadcastAxid(data, gradient) -> None:
#    # Store old shapes
#    tensorShape = np.array(data.shape)
#
#    # Get new shape
#    gradientShape = np.array(gradient.shape)
#
#    # Prepend ones to the shape of the smaller array
#    if len(tensorShape) < len(gradientShape):
#        tensorShape = np.pad(tensorShape, (len(gradientShape) - len(tensorShape), 0), mode='constant', constant_values=1)
#    elif len(tensorShape) > len(gradientShape):
#        gradientShape = np.pad(gradientShape, (len(tensorShape) - len(gradientShape), 0), mode='constant', constant_values=1)
#
#    # Find broadcasted axes
#    tensorBroadcastAxis = np.where(tensorShape != gradientShape)[0]
#
#    # Change broadcastAxis variables to None if they're empty
#    if tensorBroadcastAxis.size == 0:
#        tensorBroadcastAxis = None
#
#    return tensorBroadcastAxis


def addForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.add(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(addBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def addBackward(tensor1: Tensor, tensor2: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        if tensor1.gradientFunc:
            tensor1.gradientFunc(gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        if tensor2.gradientFunc:
            tensor2.gradientFunc(gradient)


def subtractForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.subtract(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(subtractBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def subtractBackward(tensor1: Tensor, tensor2: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        if tensor1.gradientFunc:
            tensor1.gradientFunc(gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        if tensor2.gradientFunc:
            tensor2.gradientFunc(np.negative(gradient))


def multiplyForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.multiply(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(multiplyBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)



def multiplyBackward(tensor1: Tensor, tensor2: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(tensor2.data, gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.multiply(tensor1.data, gradient)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def divideForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.divide(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(divideBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def divideBackward(tensor1: Tensor, tensor2: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.divide(gradient, tensor2.data)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.negative(np.divide(np.multiply(tensor1.data, gradient), np.power(tensor2.data, 2)))
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def matmulForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.matmul(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(matmulBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def matmulBackward(tensor1: Tensor, tensor2: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        if len(tensor1.data.shape) > 2 or len(tensor2.data.shape) > 2:
            tensor1.gradient = np.matmul(gradient, np.transpose(tensor2.data, axes=(0, 2, 1)))
        else:
            tensor1.gradient = np.matmul(gradient, np.transpose(tensor2.data))

        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        if len(tensor1.data.shape) > 2 or len(tensor2.data.shape) > 2:
            tensor2.gradient = np.matmul(np.transpose(tensor1.data, axes=(0, 2, 1)), gradient)
        else:
            tensor2.gradient = np.matmul(np.transpose(tensor1.data), gradient)

        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def dotForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.dot(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(dotBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def dotBackward(tensor1: Tensor, tensor2: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(tensor2.data, gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.negative(np.multiply(tensor1.data, gradient))
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def powerForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.power(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(powerBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def powerBackward(tensor1: Tensor, tensor2: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(np.multiply(tensor2.data, np.power(tensor1.data, (np.subtract(tensor2.data, 1)))), gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.multiply(np.multiply(np.log(tensor1.data), np.power(tensor1.data, tensor2.data)), gradient)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


#
# Single Tensor
#


def squareForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.square(tensor.data, *args, **kwargs)

    gradientFunc = partial(squareBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def squareBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(np.multiply(tensor.data, 2.0), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def sqrtForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.sqrt(tensor.data, *args, **kwargs)

    gradientFunc = partial(sqrtBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def sqrtBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.divide(gradient, np.multiply(2, np.sqrt(tensor.data)))
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def logForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.log(tensor.data, *args, **kwargs)

    gradientFunc = partial(logBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def logBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply((np.divide(1, tensor.data)), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def expForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.exp(tensor.data, *args, **kwargs)

    gradientFunc = partial(expBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def expBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(np.exp(tensor.data), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def sinForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.sin(tensor.data, *args, **kwargs)

    gradientFunc = partial(sinBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def sinBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(np.cos(tensor.data), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def cosForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.cos(tensor.data, *args, **kwargs)

    gradientFunc = partial(cosBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def cosBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.negative(np.multiply(np.sin(tensor.data), gradient))
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def tanForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.tan(tensor.data, *args, **kwargs)

    gradientFunc = partial(tanBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def tanBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply((np.divide(1, np.power(np.cos(tensor.data), 2))), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def sinhForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.sinh(tensor.data, *args, **kwargs)

    gradientFunc = partial(sinhBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def sinhBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(np.cosh(tensor.data), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def coshForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.cosh(tensor.data, *args, **kwargs)

    gradientFunc = partial(coshBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def coshBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor and tensor.requireGradient:
        tensor.gradient = np.multiply(np.sinh(tensor.data), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def tanhForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.tanh(tensor.data, *args, **kwargs)

    gradientFunc = partial(tanhBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def tanhBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply((np.divide(1, np.power(np.cosh(tensor.data), 2))), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def absForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.abs(tensor.data, *args, **kwargs)

    gradientFunc = partial(absBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def absBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(np.sign(tensor.data), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


#
# Signs
#


def signForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.sign(tensor.data, *args, **kwargs)

    gradientFunc = partial(signBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def signBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor and tensor.requireGradient:
        tensor.gradient = np.add(tensor.gradient, np.multiply(np.sign(tensor.data), gradient))
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def positiveForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.positive(tensor.data, *args, **kwargs)

    gradientFunc = partial(positiveBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def positiveBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(np.positive(tensor.data), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def negativeForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.negative(tensor.data, *args, **kwargs)

    gradientFunc = partial(negativeBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def negativeBackward(tensor: Tensor, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(np.negative(tensor.data), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


#
# Compare
#


def equalForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.equal(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(equalBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def equalBackward(tensor1: Tensor, tensor2: Tensor, bools: np.ndarray, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(bools, gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.multiply(bools, gradient)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def notEqualForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.not_equal(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(notEqualBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def notEqualBackward(tensor1: Tensor, tensor2: Tensor, bools: np.ndarray, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(bools, gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.multiply(bools, gradient)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def lessForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.less(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(lessBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def lessBackward(tensor1: Tensor, tensor2: Tensor, bools: np.ndarray, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(bools, gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.multiply(bools, gradient)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def lessEqualForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.less_equal(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(lessEqualBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def lessEqualBackward(tensor1: Tensor, tensor2: Tensor, bools: np.ndarray, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(bools, gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.multiply(bools, gradient)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def greaterForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.greater(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(greaterBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def greaterBackward(tensor1: Tensor, tensor2: Tensor, bools: np.ndarray, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(bools, gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.multiply(bools, gradient)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def greaterEqualForward(tensor1: Tensor, tensor2: Tensor, *args, **kwargs) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.greater_equal(tensor1.data, tensor2.data, *args, **kwargs)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(greaterEqualBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def greaterEqualBackward(tensor1: Tensor, tensor2: Tensor, bools: np.ndarray, gradient: np.ndarray, *args, **kwargs) -> None:
    if tensor1.requireGradient:
        #gradientForTensor1 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor1, gradientForTensor1)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor1 = np.sum(gradientForTensor1, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor1.gradient = np.multiply(bools, gradient)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        #gradientForTensor2 = np.copy(gradient)

        #tensorBroadcastAxis = getbroadcastAxid(tensor2, gradientForTensor2)
        #if tensorBroadcastAxis is not None:
        #    gradientForTensor2 = np.sum(gradientForTensor2, axis=tuple(tensorBroadcastAxis), keepdims=True)

        tensor2.gradient = np.multiply(bools, gradient)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


#
# Shaping
#


def flattenForward(tensor: Tensor) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.reshape(tensor.data, newshape=(-1))

    gradientFunc = partial(flattenBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def flattenBackward(tensor: Tensor, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.reshape(gradient, newshape=tensor.shape)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def reshapeForward(tensor: Tensor, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.reshape(tensor.data, *args, **kwargs)

    gradientFunc = partial(reshapeBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def reshapeBackward(tensor: Tensor, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.reshape(gradient, newshape=tensor.shape)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


#
# Broadcasting
#

def repeatForward(tensor: Tensor, repeats: ArrayLike, axis: int = None) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.repeat(tensor.data, repeats=repeats, axis=axis)

    gradientFunc = partial(repeatBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def repeatBackward(tensor, repeats, axis, gradient) -> None:
    if tensor.requireGradient:
        if axis is None:
            sum_axis = tuple(range(gradient.ndim)[::-repeats])
            counts = np.prod(repeats)
        else:
            sum_axis = axis
            counts = repeats

        grad = np.sum(gradient, axis=sum_axis, keepdims=True)
        grad = np.divide(grad, counts)
        tensor.gradient = np.broadcast_to(grad, tensor.shape)

        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def tileForward(tensor: Tensor, reps: ArrayLike) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.tile(tensor.data, reps=reps)

    gradientFunc = partial(tileBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def tileBackward(tensor, reps, gradient) -> None:
    if tensor.requireGradient:
        reshaped = np.reshape(gradient, tensor.shape + reps)
        axis = tuple(range(tensor.ndim, gradient.ndim))
        tensor.gradient = np.sum(reshaped, axis=axis)

        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def concatenateForward(tensors: Tensor, axis=0, out=None, dtype=None, casting='same_kind') -> Tensor:
    tensors = [checkTensor(tensor) for tensor in tensors]

    data = np.concatenate([tensor.data for tensor in tensors], axis=axis, out=out, dtype=dtype, casting=casting)

    requireGradient = any(tensor.requireGradient for tensor in tensors)
    if requireGradient:
        shapes = [tensor.shape for tensor in tensors]
        gradfunc = partial(concatenateBackward, tensors, shapes, axis, out, dtype, casting)
        return Tensor(data, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(data, requireGradient=False, gradientFunc=None)


def concatenateBackward(tensors: Tensor, shapes, axis=0, out=None, dtype=None, casting='same_kind', gradient: np.ndarray = None) -> None:
    grads = np.split(gradient, np.cumsum([shape[axis] for shape in shapes[:-1]]), axis=axis)
    for tensor, grad in zip(tensors, grads):
        if tensor.requireGradient:
            tensor.gradient = grad
            if tensor.gradientFunc:
                tensor.gradientFunc(tensor.gradient)


def hstackForward(tensors: Tensor, dtype=None, casting='same_kind') -> Tensor:
    return concatenateForward(tensors, axis=1, out=None, dtype=dtype, casting=casting)


def vstackForward(tensors: Tensor, dtype=None, casting='same_kind') -> Tensor:
    return concatenateForward(tensors, axis=0, out=None, dtype=dtype, casting=casting)


def dstackForward(tensors: Tensor, dtype=None, casting='same_kind') -> Tensor:
    return concatenateForward(tensors, axis=2, out=None, dtype=dtype, casting=casting)


def splitForward(tensor: Tensor, indices_or_sections, axis=0) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.split(tensor.data, indices_or_sections, axis)

    if tensor.requireGradient:
        gradfunc = partial(splitBackward, tensor, axis)
        return [Tensor(datum, requireGradient=True, gradientFunc=gradfunc) for datum in data]

    return [Tensor(datum, requireGradient=False, gradientFunc=None) for datum in data]


def splitBackward(tensor: Tensor, axis=0, gradient=None) -> None:
    gradient = np.concatenate(gradient, axis=axis)
    if tensor.requireGradient:
        tensor.gradient = gradient
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def hsplitForward(tensor: Tensor, indices_or_sections) -> Tensor:
    return splitForward(tensor, indices_or_sections, axis=1)


def vsplitForward(tensor: Tensor, indices_or_sections) -> Tensor:
    return splitForward(tensor, indices_or_sections, axis=0)


def dsplitForward(tensor: Tensor, indices_or_sections) -> Tensor:
    return splitForward(tensor, indices_or_sections, axis=2)


#
# Reduce
#


def sumForward(tensor: Tensor, axis=None, dtype=None, keepdims=False, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.sum(tensor.data, axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)

    gradientFunc = partial(sumBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def sumBackward(tensor: Tensor, gradient) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.broadcast_to(gradient.T, tensor.shape)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def prodForward(tensor: Tensor, axis=None, dtype=None, keepdims=False) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.prod(tensor, axis=axis, dtype=dtype, keepdims=keepdims)

    gradientFunc = partial(prodBackward, tensor, axis, type, keepdims) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def prodBackward(tensor: Tensor, axis, dtype, keepdims, gradient) -> None:
    if tensor.requireGradient:
        tensorNoneZero = np.where(tensor.data != 0, tensor.data, 1)
        tensor.gradient = np.multiply(gradient, np.divide(np.prod(tensor.data, axis=axis, dtype=dtype, keepdims=keepdims), tensorNoneZero))
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


#
# Minimum/Maximum etc
#


def maximumForward(tensor1: Tensor, tensor2: Tensor, out=None, where=True, casting='same_kind', order='k', dtype=None, subhok=True) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.maximum(tensor1.data, tensor2.data, out=out, where=where, casting=casting, order=order, dtype=dtype, subhok=subhok)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(maximumBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def maximumBackward(tensor1: Tensor, tensor2: Tensor, data: np.ndarray, gradient: np.ndarray) -> None:
    if tensor1.requireGradient:
        mask = (tensor1.data == data)
        tensor1.gradient = np.multiply(gradient, mask)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        mask = (tensor2.data == data)
        tensor2.gradient = np.multiply(gradient, mask)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def minimumForward(tensor1: Tensor, tensor2: Tensor, out=None, where=True, casting='same_kind', order='k', dtype=None, subhok=True) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.minimum(tensor1.data, tensor2.data, out=out, where=where, casting=casting, order=order, dtype=dtype, subhok=subhok)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(minimumBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def minimumBackward(tensor1: Tensor, tensor2: Tensor, data: np.ndarray, gradient: np.ndarray) -> None:
    if tensor1.requireGradient:
        mask = (tensor1.data == data)
        tensor1.gradient = np.multiply(gradient, mask)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        mask = (tensor2.data == data)
        tensor2.gradient = np.multiply(gradient, mask)
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


#
# Min/Max etc
#


def maxForward(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.max(tensor.data, axis=axis, keepdims=keepdims)

    if tensor.requireGradient:
        mask = (tensor.data == np.broadcast_to(data, tensor.shape))
        gradfunc = partial(maxBackward, tensor, mask)
        return Tensor(data, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(data, requireGradient=False, gradientFunc=None)


def maxBackward(tensor: Tensor, mask: np.ndarray, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(mask, gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def minForward(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.min(tensor.data, axis=axis, keepdims=keepdims)

    if tensor.requireGradient:
        mask = (tensor.data == np.broadcast_to(data, tensor.shape))
        gradfunc = partial(minBackward, tensor, mask)
        return Tensor(data, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(data, requireGradient=False, gradientFunc=None)


def minBackward(tensor: Tensor, mask: np.ndarray, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(mask, gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def meanForward(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.mean(tensor.data, axis=axis, keepdims=keepdims)

    if tensor.requireGradient:
        if axis is None:
            divisor = np.prod(tensor.shape)
        elif isinstance(axis, int):
            divisor = np.prod(tensor.shape[axis])
        else:
            divisor = np.prod([tensor.shape[i] for i in axis])

        gradfunc = partial(meanBackward, tensor, divisor)
        return Tensor(data, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(data, requireGradient=False, gradientFunc=None)


def meanBackward(tensor: Tensor, divisor: np.ndarray, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.divide(gradient, divisor)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def varForward(tensor: Tensor, axis=None, ddof=0, keepdims=False) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.var(tensor.data, axis=axis, ddof=ddof, keepdims=keepdims)

    if tensor.requireGradient:
        diff = np.subtract(tensor.data, np.mean(tensor.data, axis=axis, keepdims=keepdims))

        if axis is None:
            divisor = np.prod(tensor.shape)
        elif isinstance(axis, int):
            divisor = np.prod(tensor.shape[axis])
        else:
            divisor = np.prod([tensor.shape[i] for i in axis])

        gradfunc = partial(varBackward, tensor, divisor, diff)
        return Tensor(data, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(data, requireGradient=False, gradientFunc=None)


def varBackward(tensor: Tensor, divisor: np.ndarray, diff: np.ndarray, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(np.multiply(np.divide(2.0, divisor), diff), gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def stdForward(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.std(tensor.data, axis=axis, keepdims=keepdims)

    if tensor.requireGradient:
        diff = np.subtract(tensor.data, np.mean(tensor.data, axis=axis, keepdims=keepdims))

        if axis is None:
            divisor = np.prod(tensor.shape)
        elif isinstance(axis, int):
            divisor = np.prod(tensor.shape[axis])
        else:
            divisor = np.prod([tensor.shape[i] for i in axis])

        gradfunc = partial(stdBackward, tensor, divisor, diff)
        return Tensor(data, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(data, requireGradient=False, gradientFunc=None)


def stdBackward(tensor: Tensor, divisor: np.ndarray, diff: np.ndarray, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.multiply(gradient, np.divide(diff, np.multiply(divisor, tensor.data)))
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


#
# Others
#


def padForward(tensor: Tensor, pad_with, mode='constant', constant_values=0) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.pad(tensor.data, pad_with=pad_with, mode=mode, constant_values=constant_values)

    gradientFunc = partial(padBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def padBackward(tensor: Tensor, pad_with, gradient: np.ndarray) -> None:
    if tensor and tensor.requireGradient:
        slices = tuple(slice(pad[0], -pad[1] if pad[1] != 0 else None) for pad in pad_with)
        tensor.gradient = np.add(tensor.gradient, gradient[slices])
        if tensor.requireGradient:
            tensor.gradientFunc(tensor.gradient)


def insertForward(tensor: Tensor, values: Tensor, index: ArrayLike) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    values = values if isinstance(values, Tensor) else Tensor(tensor2)

    data = np.insert(tensor.data, index, values.data)

    gradientFunc = partial(insertBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def insertBackward(tensor: Tensor, values: Tensor, index: ArrayLike, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.delete(gradient, index)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)

    if values.requireGradient:
        values.gradient = gradient[index]
        if values.gradientFunc:
            values.gradientFunc(values.gradient)


def transposeForward(tensor: Tensor) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.transpose(tensor.data)

    gradientFunc = partial(transposeBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def transposeBackward(tensor: Tensor, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.transpose(gradient)
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def whereForward(condition, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    data = np.where(condition, tensor1.data, tensor2.data)

    requireGradient = tensor1.requireGradient or tensor2.requireGradient
    gradientFunc = partial(whereBackward, tensor1, tensor2) if requireGradient else None

    return Tensor(data, requireGradient=requireGradient, gradientFunc=gradientFunc)


def whereBackward(condition, tensor1: Tensor, tensor2: Tensor, gradient: np.ndarray) -> None:
    if tensor1.requireGradient:
        tensor1.gradient = np.multiply(gradient, condition)
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        tensor2.gradient = np.multiply(gradient, np.logical_not(condition))
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


def cumsumForward(tensor: Tensor, axis, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.cumsum(tensor.data, axis, *args, **kwargs)

    gradientFunc = partial(cumsumBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def cumsumBackward(tensor: Tensor, axis, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.cumsum(gradient, -axis)[::-1]
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def cumprodForward(tensor: Tensor, axis, *args, **kwargs) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    data = np.cumprod(tensor.data, axis, *args, **kwargs)

    gradientFunc = partial(cumprodBackward, tensor) if tensor.requireGradient else None
    return Tensor(data, requireGradient=tensor.requireGradient, gradientFunc=gradientFunc)


def cumprodBackward(tensor: Tensor, axis, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = np.divide(gradient, np.comprod(tensor.data))
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


#
# Not working correctly
#


def asStridedForward(tensor: Tensor, shape=None, strides=None, subok=False) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    patches = np.as_strided(tensor.data, shape=shape, strides=strides, subok=subok)

    if tensor.requireGradient:
        gradfunc = partial(asStridedBackward, tensor)
        return Tensor(data=patches, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(data=patches, requireGradient=False, gradientFunc=None)


def asStridedBackward(tensor: Tensor, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = gradient.sum(tuple(np.arange(gradient.ndim - tensor.ndim)))
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def slidingWindowForward(tensor: Tensor, window_shape=None, axis=None, *, subok=False, writeable=True) -> Tensor:
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    patches = np.sliding_window_view(tensor.data, window_shape=window_shape, axis=axis, subok=subok, writeable=writeable)

    if tensor.requireGradient:
        gradfunc = partial(slidingWindowBackward, tensor)
        return Tensor(data=patches, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(data=patches, requireGradient=False, gradientFunc=None)


def slidingWindowBackward(tensor: Tensor, gradient: np.ndarray) -> None:
    if tensor.requireGradient:
        tensor.gradient = gradient.sum(tuple(np.range(gradient.ndim - tensor.data.ndim)))
        if tensor.gradientFunc:
            tensor.gradientFunc(tensor.gradient)


def einsumForward(tensor1: Tensor, tensor2: Tensor, optimize=False) -> Tensor:
    tensor1 = tensor1 if isinstance(tensor1, Tensor) else Tensor(tensor1)
    tensor2 = tensor2 if isinstance(tensor2, Tensor) else Tensor(tensor2)

    einsums = np.einsum('bihwkl,oikl->bohw', tensor1.data, tensor2.data, optimize=optimize)

    if tensor1.requireGradient or tensor2.requireGradient:
        gradfunc = partial(einsumBackward, tensor1, tensor2, optimize)
        return Tensor(einsums, requireGradient=True, gradientFunc=gradfunc)

    return Tensor(einsums, requireGradient=False, gradientFunc=None)


def einsumBackward(tensor1: Tensor, tensor2: Tensor, optimize, gradient: np.ndarray) -> None:
    if tensor1.requireGradient:
        tensor1.gradient = np.as_strided(gradient, shape=(*tensor1.data.shape, *tensor2.data.shape[-2:]), strides=(*tensor1.data.strides, 0, 0))
        if tensor1.gradientFunc:
            tensor1.gradientFunc(tensor1.gradient)

    if tensor2.requireGradient:
        tensor2.gradient = np.as_strided(gradient, shape=(*tensor2.data.shape[:-2], *tensor1.data.shape[-2:]), strides=(0, 0, *tensor1.data.strides[-2:]))
        if tensor2.gradientFunc:
            tensor2.gradientFunc(tensor2.gradient)


#
# Mapping from Numpy to Tensor
#


ufuncMap = {
    np.add: addForward,
    np.subtract: subtractForward,
    np.multiply: multiplyForward,
    np.divide: divideForward,
    np.matmul: matmulForward,
    np.power: powerForward,
    np.square: squareForward,
    np.sqrt: sqrtForward,
    np.log: logForward,
    np.exp: expForward,
    np.sin: sinForward,
    np.cos: cosForward,
    np.cos: tanForward,
    np.sinh: sinhForward,
    np.cosh: coshForward,
    np.tanh: tanhForward,
    np.abs: absForward,
    np.sign: signForward,
    np.positive: positiveForward,
    np.negative: negativeForward,
    np.equal: equalForward,
    np.not_equal: notEqualForward,
    np.less: lessForward,
    np.less_equal: lessEqualForward,
    np.greater: greaterForward,
    np.greater_equal: greaterEqualForward,
    np.maximum: maximumForward,
    np.minimum: minimumForward
}

funcMap = {
    np.dot: dotForward,
    np.sum: sumForward,
    np.prod: prodForward,
    np.repeat: repeatForward,
    np.tile: tileForward,
    np.max: maxForward,
    np.min: minForward,
    np.mean: meanForward,
    np.var: varForward,
    np.std: stdForward,
    np.reshape: reshapeForward,
    np.transpose: transposeForward,
    np.concatenate: concatenateForward,
    np.hstack: hstackForward,
    np.vstack: vstackForward,
    np.dstack: dstackForward,
    np.split: splitForward,
    np.hsplit: hsplitForward,
    np.vsplit: vsplitForward,
    np.dsplit: dsplitForward,
    np.pad: padForward,
    np.insert: insertForward,
    np.where: whereForward,
    np.cumsum: cumsumForward,
    np.cumprod: cumprodForward,
    np.einsum: einsumForward
}