import numpy as np
from numpy.typing import NDArray
from .module import Module


class Parallel(Module):
    """
    a list for layers to be called in parallel
    incorporates the forward and backward pass
    """
    __slots__ = ['shapes', 'batchSize', 'outputShapes', 'slices', 'splits', 'inputs']

    def __init__(self, layers: list | None = None, splits: list | None = None) -> None:
        super().__init__(layers)
        self.shapes = None
        self.splits = splits

    def forward(self, *inputs: NDArray) -> NDArray:
        """
        calls all layers in parallel and stacks the outputs
        """
        self.batchSize = inputs[0].shape[0]

        # stacking inputs for iterating
        if self.splits is not None:
            inputs = [inputs[0][:,one:two] for one, two in zip(self.splits,self.splits[1:])]
            self.inputs = inputs
        elif len(inputs) == 1:
            inputs = [inputs[0]] * len(self)
        if len(inputs) != len(self):
            raise TypeError('number of input elements must be equal to Layers/Modules or one')
        outputs = []
        slices = [0] # used for slicing up incoming gradient
        self.outputShapes = []

        # iterating over layers with inputs
        for layer, input in zip(self, inputs):
            output = layer(input)
            self.outputShapes.append(output.shape[1:])
            outputs.append(output.reshape(self.batchSize,-1))
            slices.append(outputs[-1].shape[1])

        # slices for splitting gradients
        self.slices = np.cumsum(slices)

        return np.hstack(([output for output in outputs]))

    def __call__(self, *inputs: NDArray) -> NDArray:
        """
        needed to overwrite the call method, in order to have multiple inputs
        """
        return self.forward(*inputs)

    def backward(self, gradient: NDArray) -> NDArray:
        """
        calls all layers in parallel in reverse
        """
        gradients = []
        for layer, start, stop, shape in zip(self, self.slices, self.slices[1:], self.outputShapes):
            grad = layer.backward(gradient[:,start:stop].reshape(self.batchSize, *shape))
            gradients.append(grad)
        return np.array(gradients)
