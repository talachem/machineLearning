import numpy as np
from .module import Module, Sequential, Parallel
from .optim import Optimizer
from .loss import LossFunction
from .scheduler import Scheduler
from data.dataLoader import DataLoader
from utility.progressbar import Progressbar
from .layer import Layer
from importlib import import_module
from inspect import signature


class Network(object):
    """
    this class uses modules together with loss functions
    and optimizers to handel the full neural network stack
    """
    def __init__(self):
        self.name = self.__class__.__name__
        self.module = Sequential()
        self.lossFunc = None
        self.optim = None
        self.scheduler = None

    @property
    def qualifiedName(self) -> tuple:
        return self.__class__.__module__, self.__class__.__name__

    def setComponent(self, component: Optimizer | LossFunction | Scheduler) -> None:
        if isinstance(component, Optimizer):
            self.optim = component
        elif isinstance(component, LossFunction):
            self.lossFunc = component
        elif isinstance(component, Scheduler):
            self.scheduler = component
        else:
            raise TypeError("The given component is not a valid type")

    def append(self, layer: Layer, mode: str = 's') -> None:
        if mode == 's':
            if isinstance(self.module[-1], Parallel) and len(self.module[-1]) <= 1:
                toPop = self.module[-1][0]
                self.module.pop(-1)
                self.module.append(toPop)
            self.module.append(layer)
        if mode == 'p':
            if isinstance(self.module[-1], Parallel):
                self.module[-1].append(layer)
            else:
                self.module.append(Parallel())
                self.module[-1].append(layer)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        this makes using this class more convenient
        """
        return self.forward(input)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.module.forward(input)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.module.backward(gradient)

    def trainMode(self) -> None:
        """
        sets every layer in the module into train mode
        """
        self.module.train()

    def evalMode(self) -> None:
        """
        sets every layer in the module into eval mode
        """
        self.module.eval()

    def __str__(self):
        printString = ""
        printString += self.optim.name + ": " + str(self.optim.learningRate) + "\n"
        if self.scheduler is not None:
            printString += self.scheduler.name + "\n"
        printString += self.lossFunc.name + "\n"
        printString += str(self.module)

    def train(self, data: DataLoader, epochs: int) -> None:
        metrics = NetworkObservables(epochs)
        epochs = epochs
        for i in range(epochs):
            data.trainMode()
            self.trainMode()
            length = len(data)
            bar = Progressbar(f'epoch {str(i+1).zfill(len(str(epochs)))}/{epochs}', length, 55)
            losses = []
            for item in data:
                inputs = item['data']
                labels = item['labels']
                prediction = network(inputs)
                losses.append(self.loss(prediction, labels))
                gradient = self.loss.backward()
                self.optim.step(gradient)
                bar.step()
            data.evalMode()
            self.evalMode()
            accuracies = []
            valLosses = []
            for item in data:
                inputs = item['data']
                labels = item['labels']
                prediction = network(inputs)
                valLosses.append(self.loss(prediction, labels))
                accuracies.append(np.sum(prediction.argmax(1) == labels.argmax(1)) / len(prediction))
                bar.step()

            metrics.update('losses', np.mean(losses))
            metrics.update('validation', np.mean(valLosses))
            metrics.update('accuracy', np.mean(accuracies))
            metrics.update('learningRate', self.optim.learningRate)
            metrics.print()
            metrics.step()
            data.fold()

            if self.scheduler is not None:
                self.scheduler.step()

    def eval(self, data: DataLoader) -> np.ndarray:
        self.evalEval()
        length = len(data)
        bar = Progressbar('evaluation', length)
        predictions = []
        for item in data:
            inputs = item['data']
            labels = item['labels']
            predictions.append(network(inputs))
            bar.step()

        return np.concatnate(predictions)

    def __getitem__(self, index):
        return self.module[index]