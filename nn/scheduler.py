import numpy as np
from abc import ABC, abstractmethod
from typing import Protocol


class Teacher(Protocol):
    @property
    def learningRate(self) -> float:
        ...

def cosineRange(startRate: float, endRate: float, epochs: int) -> np.ndarray:
    """
    This function generates a cosine shape for a given range
    """
    epoch = np.linspace(0, np.pi, epochs)
    rates = (np.cos(epoch) + 1) * 0.5 * (endRate - startRate) + startRate
    return rates


class Scheduler(ABC):
    """
    base class of learning rate scheduler
    """
    __slots__ = ['name', 'learningRate', 'teacher']

    def __init__(self, teacher: Teacher) -> None:
        self.name = self.__class__.__name__
        self.teacher = teacher
        self.learningRate = self.teacher.learningRate

    @abstractmethod
    def update(self) -> float:
        """
        this is used for calculating the new learning rate
        """
        pass

    def step(self) -> None:
        """
        this hands over the new learning rate
        """
        self.teacher.learningRate = self.update()


class ExponentialLR(Scheduler):
    """
    implementation of an exponential learning rate decay
    """
    __slots__ = ['decayRate']

    def __init__(self, teacher: Teacher, decayRate: float) -> None:
        super().__init__(teacher)
        self.decayRate = decayRate

    def update(self) -> float:
        """
        this decreases the learning rate by a constant factor
        """
        self.learningRate = self.learningRate * self.decayRate
        return self.learningRate


class SteppedLR(Scheduler):
    """
    implementation of stepped learning rate
    """
    __slots__ = ['decayRate', 'stepSize', '_steps']

    def __init__(self, teacher: Teacher, decayRate: float, stepSize: int) -> None:
        super().__init__(teacher)
        self.decayRate = decayRate
        self.stepSize = stepSize
        self._steps = 0

    def update(self) -> float:
        """
        counts steps and if at a certain milestone and
        decreases learning rate by a constant factor
        """
        if self._steps % self.stepSize == 0 and self._steps != 0:
            self.learningRate = self.learningRate * self.decayRate
        self._steps += 1
        return self.learningRate


class CyclicalLR(Scheduler):
    """
    learningRate is the base rate
    limitRate is the upper or lower learningRate limit
    stepsUp/stepsDown is the time it takes to go up/down
    """
    __slots__ = ['totalSteps', 'steps', 'rates']

    def __init__(self, teacher: Teacher, learningRateScale: float, stepsUp: int, stepsDown: int, shape: str = 'zickzack') -> None:
        super().__init__(teacher)
        self.totalSteps = stepsUp + stepsDown
        self.learningRateScale = learningRateScale
        self.stepsUp = stepsUp
        self.stepsDown = stepsDown
        self.steps = 0
        self.shape = shape
        limitRate = self.learningRate * learningRateScale

        # creating linearly increasing/decreasing learning rate
        if shape == 'zickzack':
            if self.learningRate > limitRate:
                self.rates = np.concatenate((np.arange(self.learningRate, limitRate, (limitRate-self.learningRate)/stepsDown), np.arange(limitRate, self.learningRate, (self.learningRate-limitRate)/stepsUp)))
            if self.learningRate < limitRate:
                self.rates = np.concatenate((np.arange(self.learningRate, limitRate, (limitRate-self.learningRate)/stepsUp), np.arange(limitRate, self.learningRate, (self.learningRate-limitRate)/stepsDown)))

        # creating a stepped style increasing/decreasing learning rate
        elif shape == 'stepped':
            if self.learningRate < limitRate:
                self.rates = np.array([self.learningRate] * stepsUp + [limitRate] * stepsDown)
            if self.learningRate > limitRate:
                self.rates = np.array([self.learningRate] * stepsDown + [limitRate] * stepsUp)

        # creating smoothly increasing/decreasing learning rate
        elif shape == 'cosine':
            if self.learningRate < limitRate:
                ratesStart = cosineRange(self.learningRate, limitRate, stepsUp)
                ratesEnd = cosineRange(limitRate, self.learningRate, stepsDown)
            if self.learningRate > limitRate:
                ratesStart = cosineRange(self.learningRate, limitRate, stepsDown)
                ratesEnd = cosineRange(limitRate, self.learningRate, stepsUp)
            self.rates = np.concatenate((ratesStart, ratesEnd))
        else:
            raise ValueError(f'{shape} is not an option for shape')

    def update(self) -> float:
        """
        this steps through prepared learning rate arrays
        """
        learningRate = self.rates[self.steps]
        if self.steps == self.totalSteps - 1:
            self.steps = 0
        else:
            self.steps += 1
        return learningRate
