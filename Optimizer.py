import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor

from Loss import MeanSquareError, Loss
from Layer import Layer, Linear, Activation, Sigmoid
from NeuralNet import NeuralNet

Func = Callable[[Tensor], Tensor]

class Optimizer:
    def step(self, net: NeuralNet) ->None:
        raise NotImplementedError
    