import numpy as np
from .layer import Layer
from .weights import Weights


class Hopfield(Layer):
    def __init__(self):
        super().__init__()
