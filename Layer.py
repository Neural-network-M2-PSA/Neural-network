import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor

from Loss import MeanSquareError, Loss

Func = Callable[[Tensor], Tensor]

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

def forward(self, inputs: Tensor) -> Tensor:

    self.inputs = inputs
    #z = np.transpose(self.params["w"])@self.inputs + self.params["b"]
    #print('z = ',z)

def backward(self, grad: Tensor) -> Tensor:

    a = Tensor
    b = MeanSquareError()
    w = MeanSquareError()
    self.grads["w"] = w.grad(self.params["w"], self.inputs)
    self.grads["b"] = b.grad(self.params["b"], self.inputs)
    A = [self.grads["w"], self.grads["b"]]
    print("A : ", A)
    return A
