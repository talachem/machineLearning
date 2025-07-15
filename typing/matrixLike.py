from typing_extensions import Generic, NewType, Sequence, TypeVarTuple, TypeVar
import numpy as np
from numpy.typing import ArrayLike, NDArray

DType = TypeVar("DType")
Shape = TypeVarTuple("Shape")
Batch = NewType("Batch", bound=int)
Channels = NewType("Channels", bound=int)
Features = NewType("Features", bound=int)
Height = NewType("Height", bound=int)
Width = NewType("Width", bound=int)
Timesteps = NewType("Timesteps", bound=int)



class MatrixLike(np.ndarray, Generic[*Shape]):
    """
    assignes size parameters to numpy arrays
    """
    ...

class ImageLike[Batch, Channels, Height, Width](MatrixLike):
    """
    CNN input
    """
    ...

class FeaturesLike[Batch, Features](MatrixLike):
    """
    Dense input
    """
    ...
