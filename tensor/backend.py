from abc import ABC, abstractmethod
import numpy as np


class BackendInterface(ABC):
    # init

    @abstractmethod
    def array(self, x):
        raise NotImplementedError()

    @abstractmethod
    def copy(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def zeros(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def ones(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def zeros_like(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def ones_like(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def arange(self, *args, **kwargs):
        raise NotImplementedError()

    # double tensor

    @abstractmethod
    def add(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def subtract(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def multiply(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def divide(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def matmul(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def dot(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def power(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    # single tensor

    @abstractmethod
    def square(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def sqrt(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def log(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def exp(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def sin(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def cos(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def tan(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def sinh(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def cosh(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def tanh(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def abs(self, x, *args, **kwargs):
        raise NotImplementedError()

    # signs

    @abstractmethod
    def sign(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def positive(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def negative(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def negative(self, x, *args, **kwargs):
        raise NotImplementedError()

    # Compare

    @abstractmethod
    def equal(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def not_equal(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def less(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def less_equal(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def greater(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def greater_equal(self, x, y, *args, **kwargs):
        raise NotImplementedError()

    # logic

    @abstractmethod
    def logical_and(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def logical_or(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def logical_xor(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def logical_not(self, x, *args, **kwargs):
        raise NotImplementedError()

    # shaping

    @abstractmethod
    def flatten(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def reshape(self, x, *args, **kwargs):
        raise NotImplementedError()

    # broadcasting

    @abstractmethod
    def broadcast_to(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def repeat(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def tile(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def concatenate(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def split(self, x, *args, **kwargs):
        raise NotImplementedError()

    # reduce

    @abstractmethod
    def sum(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def prod(self, x, *args, **kwargs):
        raise NotImplementedError()

    # min/max etc

    @abstractmethod
    def max(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def min(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def mean(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def var(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def std(self, x, *args, **kwargs):
        raise NotImplementedError()

    # others

    @abstractmethod
    def pad(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def insert(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def transpose(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def where(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def cumsum(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def cumprod(self, x, *args, **kwargs):
        raise NotImplementedError()

    # not working yet

    @abstractmethod
    def as_strided(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def sliding_window_view(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def einsum(self, subscript, x, y, *args, **kwargs):
        raise NotImplementedError()


class NumpyBackend(BackendInterface):
    import numpy as np
    # init

    def array(self, x):
        return self.np.array(x)

    def copy(self, x, *args, **kwargs):
        return self.np.copy(x, *args, **kwargs)

    def zeros(self, x, *args, **kwargs):
        return self.np.zeros(x, *args, **kwargs)

    def ones(self, x, *args, **kwargs):
        return self.np.ones(x, *args, **kwargs)

    def zeros_like(self, x, *args, **kwargs):
        return self.np.zeros_like(x, *args, **kwargs)

    def ones_like(self, x, *args, **kwargs):
        return self.np.ones_like(x, *args, **kwargs)

    def arange(self, *args, **kwargs):
        return self.np.arange(*args, **kwargs)

    # double tensor

    def add(self, x, y, *args, **kwargs):
        return self.np.add(x, y, *args, **kwargs)

    def subtract(self, x, y, *args, **kwargs):
        return self.np.subtract(x, y, *args, **kwargs)

    def multiply(self, x, y, *args, **kwargs):
        return self.np.multiply(x, y, *args, **kwargs)

    def divide(self, x, y, *args, **kwargs):
        return self.np.divide(x, y, *args, **kwargs)

    def matmul(self, x, y, *args, **kwargs):
        return self.np.matmul(x, y, *args, **kwargs)

    def dot(self, x, y, *args, **kwargs):
        return self.np.dot(x, y, *args, **kwargs)

    def power(self, x, y, *args, **kwargs):
        return self.np.power(x, y, *args, **kwargs)

    # single tensor

    def square(self, x, *args, **kwargs):
        return self.np.square(x, *args, **kwargs)

    def sqrt(self, x, *args, **kwargs):
        return self.np.sqrt(x, *args, **kwargs)

    def log(self, x, *args, **kwargs):
        return self.np.log(x, *args, **kwargs)

    def exp(self, x, *args, **kwargs):
        return self.np.exp(x, *args, **kwargs)

    def sin(self, x, *args, **kwargs):
        return self.np.sin(x, *args, **kwargs)

    def cos(self, x, *args, **kwargs):
        return self.np.cos(x, *args, **kwargs)

    def tan(self, x, *args, **kwargs):
        return self.np.tan(x, *args, **kwargs)

    def sinh(self, x, *args, **kwargs):
        return self.np.sinh(x, *args, **kwargs)

    def cosh(self, x, *args, **kwargs):
        return self.np.cosh(x, *args, **kwargs)

    def tanh(self, x, *args, **kwargs):
        return self.np.tanh(x, *args, **kwargs)

    def abs(self, x, *args, **kwargs):
        return self.np.abs(x, *args, **kwargs)

    # signs

    def sign(self, x, *args, **kwargs):
        return self.np.sign(x, *args, **kwargs)

    def positive(self, x, *args, **kwargs):
        return self.np.positive(x, *args, **kwargs)

    def negative(self, x, *args, **kwargs):
        return self.np.negative(x, *args, **kwargs)

    # compare

    def equal(self, x, y, *args, **kwargs):
        return self.np.equal(x, y, *args, **kwargs)

    def not_equal(self, x, y, *args, **kwargs):
        return self.np.not_equal(x, y, *args, **kwargs)

    def less(self, x, y, *args, **kwargs):
        return self.np.less(x, y, *args, **kwargs)

    def less_equal(self, x, y, *args, **kwargs):
        return self.np.less_equal(x, y, *args, **kwargs)

    def greater(self, x, y, *args, **kwargs):
        return self.np.greater(x, y, *args, **kwargs)

    def greater_equal(self, x, y, *args, **kwargs):
        return self.np.greater_equal(x, y, *args, **kwargs)

    # logic

    def logical_and(self, x, *args, **kwargs):
        return self.np.logical_and(x, *args, **kwargs)

    def logical_or(self, x, *args, **kwargs):
        return self.np.logical_or(x, *args, **kwargs)

    def logical_xor(self, x, *args, **kwargs):
        return self.np.logical_xor(x, *args, **kwargs)

    def logical_not(self, x, *args, **kwargs):
        return self.np.logical_not(x, *args, **kwargs)

    # shaping

    def flatten(self, x, **kwargs):
        return self.np.reshape(x, -1, **kwargs)

    def reshape(self, x, *args, **kwargs):
        return self.np.reshape(x, *args, **kwargs)

    # broadcasting

    def broadcast_to(self, x, *args, **kwargs):
        return self.np.broadcast_to(x, *args, **kwargs)

    def repeat(self, x, *args, **kwargs):
        return self.np.repeat(x, *args, **kwargs)

    def tile(self, x, *args, **kwargs):
        return self.np.tile(x, *args, **kwargs)

    def concatenate(self, x, *args, **kwargs):
        return self.np.concatenate(x, *args, **kwargs)

    def split(self, x, *args, **kwargs):
        return self.np.split(x, *args, **kwargs)

    # reduce

    def sum(self, x, *args, **kwargs):
        return self.np.sum(x, *args, **kwargs)

    def prod(self, x, *args, **kwargs):
        return self.np.prod(x, *args, **kwargs)

    # min/max etc

    def max(self, x, *args, **kwargs):
        return self.np.max(x, *args, **kwargs)

    def min(self, x, *args, **kwargs):
        return self.np.min(x, *args, **kwargs)

    def mean(self, x, *args, **kwargs):
        return self.np.mean(x, *args, **kwargs)

    def var(self, x, *args, **kwargs):
        return self.np.var(x, *args, **kwargs)

    def std(self, x, *args, **kwargs):
        return self.np.std(x, *args, **kwargs)

    # others

    def pad(self, x, *args, **kwargs):
        return self.np.pad(x, *args, **kwargs)

    def insert(self, x, *args, **kwargs):
        return self.np.insert(x, *args, **kwargs)

    def transpose(self, x, *args, **kwargs):
        return self.np.transpose(x, *args, **kwargs)

    def where(self, x, *args, **kwargs):
        return self.np.where(x, *args, **kwargs)

    def cumsum(self, x, *args, **kwargs):
        return self.np.cumsum(x, *args, **kwargs)

    def cumprod(self, x, *args, **kwargs):
        return self.np.cumprod(x, *args, **kwargs)

    # not working yet

    def as_strided(self, x, *args, **kwargs):
        return self.np.lib.stride_tricks.as_strided(x, *args, **kwargs)

    def sliding_window_view(self, x, *args, **kwargs):
        return self.np.lib.stride_tricks.sliding_window_view(x, *args, **kwargs)

    def einsum(self, subscript, x, y, *args, **kwargs):
        return self.np.einsum(subscript, x, y, *args, **kwargs)


class CupyBackend(BackendInterface):
    try:
        import cupy as cp
    except ImportError:
        pass

    # init

    def array(self, x):
        return self.cp.array(x)

    def copy(self, x, *args, **kwargs):
        return self.cp.copy(x, *args, **kwargs)

    def zeros(self, x, *args, **kwargs):
        return self.cp.zeros(x, *args, **kwargs)

    def ones(self, x, *args, **kwargs):
        return self.cp.ones(x, *args, **kwargs)

    def zeros_like(self, x, *args, **kwargs):
        return self.cp.zeros_like(x, *args, **kwargs)

    def ones_like(self, x, *args, **kwargs):
        return self.cp.ones_like(x, *args, **kwargs)

    def arange(self, *args, **kwargs):
        return self.cp.arange(*args, **kwargs)

    # double tensor

    def add(self, x, y, *args, **kwargs):
        return self.cp.add(x, y, *args, **kwargs)

    def subtract(self, x, y, *args, **kwargs):
        return self.cp.subtract(x, y, *args, **kwargs)

    def multiply(self, x, y, *args, **kwargs):
        return self.cp.multiply(x, y, *args, **kwargs)

    def divide(self, x, y, *args, **kwargs):
        return self.cp.divide(x, y, *args, **kwargs)

    def matmul(self, x, y, *args, **kwargs):
        return self.cp.matmul(x, y, *args, **kwargs)

    def dot(self, x, y, *args, **kwargs):
        return self.cp.dot(x, y, *args, **kwargs)

    def power(self, x, y, *args, **kwargs):
        return self.cp.power(x, y, *args, **kwargs)

    # single tensor

    def square(self, x, *args, **kwargs):
        return self.cp.square(x, *args, **kwargs)

    def sqrt(self, x, *args, **kwargs):
        return self.cp.sqrt(x, *args, **kwargs)

    def log(self, x, *args, **kwargs):
        return self.cp.log(x, *args, **kwargs)

    def exp(self, x, *args, **kwargs):
        return self.cp.exp(x, *args, **kwargs)

    def sin(self, x, *args, **kwargs):
        return self.cp.sin(x, *args, **kwargs)

    def cos(self, x, *args, **kwargs):
        return self.cp.cos(x, *args, **kwargs)

    def tan(self, x, *args, **kwargs):
        return self.cp.tan(x, *args, **kwargs)

    def sinh(self, x, *args, **kwargs):
        return self.cp.sinh(x, *args, **kwargs)

    def cosh(self, x, *args, **kwargs):
        return self.cp.cosh(x, *args, **kwargs)

    def tanh(self, x, *args, **kwargs):
        return self.cp.tanh(x, *args, **kwargs)

    def abs(self, x, *args, **kwargs):
        return self.cp.abs(x, *args, **kwargs)

    # signs

    def sign(self, x, *args, **kwargs):
        return self.cp.sign(x, *args, **kwargs)

    def positive(self, x, *args, **kwargs):
        return self.cp.positive(x, *args, **kwargs)

    def negative(self, x, *args, **kwargs):
        return self.cp.negative(x, *args, **kwargs)

    # compare

    def equal(self, x, y, *args, **kwargs):
        return self.cp.equal(x, y, *args, **kwargs)

    def not_equal(self, x, y, *args, **kwargs):
        return self.cp.not_equal(x, y, *args, **kwargs)

    def less(self, x, y, *args, **kwargs):
        return self.cp.less(x, y, *args, **kwargs)

    def less_equal(self, x, y, *args, **kwargs):
        return self.cp.less_equal(x, y, *args, **kwargs)

    def greater(self, x, y, *args, **kwargs):
        return self.cp.greater(x, y, *args, **kwargs)

    def greater_equal(self, x, y, *args, **kwargs):
        return self.cp.greater_equal(x, y, *args, **kwargs)
    # logic

    def logical_and(self, x, *args, **kwargs):
        return self.cp.logical_and(x, *args, **kwargs)

    def logical_or(self, x, *args, **kwargs):
        return self.cp.logical_or(x, *args, **kwargs)

    def logical_xor(self, x, *args, **kwargs):
        return self.cp.logical_xor(x, *args, **kwargs)

    def logical_not(self, x, *args, **kwargs):
        return self.cp.logical_not(x, *args, **kwargs)

    # shaping

    def flatten(self, x, **kwargs):
        return self.cp.reshape(x, -1, **kwargs)

    def reshape(self, x, *args, **kwargs):
        return self.cp.reshape(x, *args, **kwargs)

    # broadcasting

    def broadcast_to(self, x, *args, **kwargs):
        return self.cp.broadcast_to(x, *args, **kwargs)

    def repeat(self, x, *args, **kwargs):
        return self.cp.repeat(x, *args, **kwargs)

    def tile(self, x, *args, **kwargs):
        return self.cp.tile(x, *args, **kwargs)

    def concatenate(self, x, *args, **kwargs):
        return self.cp.concatenate(x, *args, **kwargs)

    def split(self, x, *args, **kwargs):
        return self.cp.split(x, *args, **kwargs)

    # reduce

    def sum(self, x, *args, **kwargs):
        return self.cp.sum(x, *args, **kwargs)

    def prod(self, x, *args, **kwargs):
        return self.cp.prod(x, *args, **kwargs)

    # min/max etc

    def max(self, x, *args, **kwargs):
        return self.cp.max(x, *args, **kwargs)

    def min(self, x, *args, **kwargs):
        return self.cp.min(x, *args, **kwargs)

    def mean(self, x, *args, **kwargs):
        return self.cp.mean(x, *args, **kwargs)

    def var(self, x, *args, **kwargs):
        return self.cp.var(x, *args, **kwargs)

    def std(self, x, *args, **kwargs):
        return self.cp.std(x, *args, **kwargs)

    # others

    def pad(self, x, *args, **kwargs):
        return self.cp.pad(x, *args, **kwargs)

    def insert(self, x, *args, **kwargs):
        return self.cp.insert(x, *args, **kwargs)

    def transpose(self, x, *args, **kwargs):
        return self.cp.negative(x, *args, **kwargs)

    def where(self, x, *args, **kwargs):
        return self.cp.where(x, *args, **kwargs)

    def cumsum(self, x, *args, **kwargs):
        return self.cp.cumsum(x, *args, **kwargs)

    def cumprod(self, x, *args, **kwargs):
        return self.cp.cumprod(x, *args, **kwargs)


    # not working yet

    def as_strided(self, x, *args, **kwargs):
        return self.cp.lib.stride_tricks.as_strided(x, *args, **kwargs)

    def sliding_window_view(self, x, *args, **kwargs):
        return self.cp.lib.stride_tricks.sliding_window_view(x, *args, **kwargs)

    def einsum(self, subscript, x, y, *args, **kwargs):
        return self.cp.einsum(subscript, x, y, *args, **kwargs)


class NumbaBackend(BackendInterface):
    from numba import jit
    # init

    @staticmethod
    @jit(nopython=True)
    def array(x):
        return np.array(x)

    @staticmethod
    @jit(nopython=True)
    def copy(x, *args, **kwargs):
        return np.copy(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def zeros(x, *args, **kwargs):
        return np.zeros(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def ones(x, *args, **kwargs):
        return np.ones(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def zeros_like(x, *args, **kwargs):
        return np.zeros_like(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def ones_like(x, *args, **kwargs):
        return np.ones_like(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def arange(*args, **kwargs):
        return np.arange(*args, **kwargs)

    # double tensor

    @staticmethod
    @jit(nopython=True)
    def add(x, y, *args, **kwargs):
        return np.add(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def subtract(x, y, *args, **kwargs):
        return np.subtract(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def multiply(x, y, *args, **kwargs):
        return np.multiply(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def divide(x, y, *args, **kwargs):
        return np.divide(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def matmul(x, y, *args, **kwargs):
        return np.matmul(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def dot(x, y, *args, **kwargs):
        return np.dot(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def power(x, y, *args, **kwargs):
        return np.power(x, y, *args, **kwargs)

    # single tensor

    @staticmethod
    @jit(nopython=True)
    def square(x, *args, **kwargs):
        return np.square(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def sqrt(x, *args, **kwargs):
        return np.sqrt(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def log(x, *args, **kwargs):
        return np.log(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def exp(x, *args, **kwargs):
        return np.exp(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def sin(x, *args, **kwargs):
        return np.sin(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def cos(x, *args, **kwargs):
        return np.cos(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def tan(x, *args, **kwargs):
        return np.tan(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def sinh(x, *args, **kwargs):
        return np.sinh(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def cosh(x, *args, **kwargs):
        return np.cosh(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def tanh(x, *args, **kwargs):
        return np.tanh(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def abs(x, *args, **kwargs):
        return np.abs(x, *args, **kwargs)

    # signs

    @staticmethod
    @jit(nopython=True)
    def sign(x, *args, **kwargs):
        return np.sign(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def positive(x, *args, **kwargs):
        return np.positive(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def negative(x, *args, **kwargs):
        return np.negative(x, *args, **kwargs)

    # compare

    @staticmethod
    @jit(nopython=True)
    def equal(x, y, *args, **kwargs):
        return np.equal(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def not_equal(x, y, *args, **kwargs):
        return np.not_equal(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def less(x, y, *args, **kwargs):
        return np.less(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def less_equal(x, y, *args, **kwargs):
        return np.less_equal(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def greater(x, y, *args, **kwargs):
        return np.greater(x, y, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def greater_equal(x, y, *args, **kwargs):
        return np.greater_equal(x, y, *args, **kwargs)

    # logic

    @staticmethod
    @jit(nopython=True)
    def logical_and(x, *args, **kwargs):
        return np.logical_and(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def logical_or(x, *args, **kwargs):
        return np.logical_or(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def logical_xor(x, *args, **kwargs):
        return np.logical_xor(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def logical_not(x, *args, **kwargs):
        return np.logical_not(x, *args, **kwargs)

    # shaping

    @staticmethod
    @jit(nopython=True)
    def flatten(x, **kwargs):
        return np.reshape(x, -1, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def reshape(x, *args, **kwargs):
        return np.reshape(x, *args, **kwargs)

    # broadcasting

    @staticmethod
    @jit(nopython=True)
    def broadcast_to(x, *args, **kwargs):
        return np.broadcast_to(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def repeat(x, *args, **kwargs):
        return np.repeat(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def tile(x, *args, **kwargs):
        return np.tile(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def concatenate(x, *args, **kwargs):
        return np.concatenate(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def split(x, *args, **kwargs):
        return np.split(x, *args, **kwargs)

    # reduce

    @staticmethod
    @jit(nopython=True)
    def sum(x, *args, **kwargs):
        return np.sum(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def prod(x, *args, **kwargs):
        return np.prod(x, *args, **kwargs)

    # min/max etc

    @staticmethod
    @jit(nopython=True)
    def max(x, *args, **kwargs):
        return np.max(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def min(x, *args, **kwargs):
        return np.min(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def mean(x, *args, **kwargs):
        return np.mean(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def var(x, *args, **kwargs):
        return np.var(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def std(x, *args, **kwargs):
        return np.std(x, *args, **kwargs)

    # others

    @staticmethod
    @jit(nopython=True)
    def pad(x, *args, **kwargs):
        return np.pad(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def insert(x, *args, **kwargs):
        return np.insert(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def transpose(x, *args, **kwargs):
        return np.transpose(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def where(x, *args, **kwargs):
        return np.where(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def cumsum(x, *args, **kwargs):
        return np.cumsum(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def cumprod(x, *args, **kwargs):
        return np.cumprod(x, *args, **kwargs)

    # not working yet

    @staticmethod
    @jit(nopython=True)
    def as_strided(x, *args, **kwargs):
        return np.lib.stride_tricks.as_strided(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def sliding_window_view( x, *args, **kwargs):
        return np.lib.stride_tricks.sliding_window_view(x, *args, **kwargs)

    @staticmethod
    @jit(nopython=True)
    def einsum(x, y, *args, **kwargs):
        return np.einsum('bihwkl,oikl->bohw', x, y, *args, **kwargs)


class PytorchBackend(BackendInterface):
    try:
        import torch
    except ImportError:
        pass
    # init

    def array(self, x, *args, **kwargs):
        return self.torch.tensor(x, *args, **kwargs)

    def copy(self, x, *args, **kwargs):
        return self.torch.clone(x)

    def zeros(self, x, *args, **kwargs):
        return self.torch.zeros(x, *args, **kwargs)

    def ones(self, x, *args, **kwargs):
        return self.torch.ones(x, *args, **kwargs)

    def zeros_like(self, x, *args, **kwargs):
        return self.torch.zeros_like(x, *args, **kwargs)

    def ones_like(self, x, *args, **kwargs):
        return self.torch.ones_like(x, *args, **kwargs)

    def arange(self, *args, **kwargs):
        return self.torch.arange(*args, **kwargs)

    # double tensor

    def add(self, x, y, *args, **kwargs):
        return self.torch.add(x, y, *args, **kwargs)

    def subtract(self, x, y, *args, **kwargs):
        return self.torch.subtract(x, y, *args, **kwargs)

    def multiply(self, x, y, *args, **kwargs):
        return self.torch.multiply(x, y, *args, **kwargs)

    def divide(self, x, y, *args, **kwargs):
        return self.torch.divide(x, y, *args, **kwargs)

    def matmul(self, x, y, *args, **kwargs):
        return self.torch.matmul(x, y, *args, **kwargs)

    def dot(self, x, y, *args, **kwargs):
        return self.torch.dot(x, y, *args, **kwargs)

    def power(self, x, y, *args, **kwargs):
        return self.torch.pow(x, y, *args, **kwargs)

    # single tensor

    def square(self, x, *args, **kwargs):
        return self.torch.square(x, *args, **kwargs)

    def sqrt(self, x, *args, **kwargs):
        return self.torch.sqrt(x, *args, **kwargs)

    def log(self, x, *args, **kwargs):
        return self.torch.log(x, *args, **kwargs)

    def exp(self, x, *args, **kwargs):
        return self.torch.exp(x, *args, **kwargs)

    def sin(self, x, *args, **kwargs):
        return self.torch.sin(x, *args, **kwargs)

    def cos(self, x, *args, **kwargs):
        return self.torch.cos(x, *args, **kwargs)

    def tan(self, x, *args, **kwargs):
        return self.torch.tan(x, *args, **kwargs)

    def sinh(self, x, *args, **kwargs):
        return self.torch.sinh(x, *args, **kwargs)

    def cosh(self, x, *args, **kwargs):
        return self.torch.cosh(x, *args, **kwargs)

    def tanh(self, x, *args, **kwargs):
        return self.torch.tanh(x, *args, **kwargs)

    def abs(self, x, *args, **kwargs):
        return self.torch.abs(x, *args, **kwargs)

    # signs

    def sign(self, x, *args, **kwargs):
        return self.torch.sign(x, *args, **kwargs)

    def positive(self, x, *args, **kwargs):
        return self.torch.positive(x, *args, **kwargs)

    def negative(self, x, *args, **kwargs):
        return self.torch.negative(x, *args, **kwargs)

    # compare

    def equal(self, x, y, *args, **kwargs):
        return self.torch.equal(x, y, *args, **kwargs)

    def not_equal(self, x, y, *args, **kwargs):
        return self.torch.not_equal(x, y, *args, **kwargs)

    def less(self, x, y, *args, **kwargs):
        return self.torch.less(x, y, *args, **kwargs)

    def less_equal(self, x, y, *args, **kwargs):
        return self.torch.less_equal(x, y, *args, **kwargs)

    def greater(self, x, y, *args, **kwargs):
        return self.torch.greater(x, y, *args, **kwargs)

    def greater_equal(self, x, y, *args, **kwargs):
        return self.torch.greater_equal(x, y, *args, **kwargs)

    # logic

    def logical_and(self, x, *args, **kwargs):
        return self.torch.logical_and(x, *args, **kwargs)

    def logical_or(self, x, *args, **kwargs):
        return self.torch.logical_or(x, *args, **kwargs)

    def logical_xor(self, x, *args, **kwargs):
        return self.torch.logical_xor(x, *args, **kwargs)

    def logical_not(self, x, *args, **kwargs):
        return self.torch.logical_not(x, *args, **kwargs)

    # shaping

    def flatten(self, x, **kwargs):
        return self.torch.reshape(x, -1, **kwargs)

    def reshape(self, x, *args, **kwargs):
        return self.torch.reshape(x, *args, **kwargs)

    # broadcasting

    def broadcast_to(self, x, *args, **kwargs):
        return self.torch.broadcast_to(x, *args, **kwargs)

    def repeat(self, x, *args, **kwargs):
        return self.torch.repeat(x, *args, **kwargs)

    def tile(self, x, *args, **kwargs):
        return self.torch.tile(x, *args, **kwargs)

    def concatenate(self, x, *args, **kwargs):
        return self.torch.cat(x, *args, **kwargs)

    def split(self, x, *args, **kwargs):
        return self.torch.split(x, *args, **kwargs)

    # reduce

    def sum(self, x, *args, **kwargs):
        return self.torch.sum(x)

    def prod(self, x, *args, **kwargs):
        return self.torch.prod(x, *args, **kwargs)

    # min/max etc

    def max(self, x, *args, **kwargs):
        return self.torch.max(x, *args, **kwargs)

    def min(self, x, *args, **kwargs):
        return self.torch.min(x, *args, **kwargs)

    def mean(self, x, *args, **kwargs):
        return self.torch.mean(x, *args, **kwargs)

    def var(self, x, *args, **kwargs):
        return self.torch.var(x, *args, **kwargs)

    def std(self, x, *args, **kwargs):
        return self.torch.std(x, *args, **kwargs)

    # others

    def pad(self, x, *args, **kwargs):
        return self.torch.pad(x, *args, **kwargs)

    def insert(self, x, *args, **kwargs):
        return self.torch.insert(x, *args, **kwargs)

    def transpose(self, x, *args, **kwargs):
        return self.torch.transpose(x, 0, 1)

    def where(self, x, *args, **kwargs):
        return self.torch.where(x, *args, **kwargs)

    def cumsum(self, x, *args, **kwargs):
        return self.torch.cumsum(x, *args, **kwargs)

    def cumprod(self, x, *args, **kwargs):
        return self.torch.cumprod(x, *args, **kwargs)

    # not working yet

    def as_strided(self, x, *args, **kwargs):
        return self.torch.as_strided(x, *args, **kwargs)

    def sliding_window_view(self, x, *args, **kwargs):
        raise NotImplementedError

    def einsum(self, subscript, x, y, *args, **kwargs):
        return self.torch.einsum(subscript, x, y, *args, **kwargs)


class TensorflowBackend(BackendInterface):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    # Implement the necessary methods here using TensorFlow's functions
    pass