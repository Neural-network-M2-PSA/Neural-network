import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor

Func = Callable[[Tensor], Tensor]

class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MeanSquareError(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:


        for i in range(len(predicted)):
            mse = mse + (predicted[i] - actual[i]) ^ 2

        return mse

    # def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
    # return ...
