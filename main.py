import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor
from Loss import Loss, MeanSquareError
Func = Callable[[Tensor], Tensor]


def test_hello():
    print('hello')
test_hello()

#Test pour voir si la loss function fonctionne
def test_loss_function():
    predicted = Tensor
    actual = Tensor
    a = MeanSquareError()
    b = MeanSquareError()
    predicted = [1, 1, 1, 1, 1, 1]
    actual = [1, 1, 2, 1, 3, 1]
    loss = a.loss(predicted, actual)
    grad = b.grad(predicted, actual)
    print('loss:', loss)
    print('grad:', grad)
    return 0

test_loss_function()

    

