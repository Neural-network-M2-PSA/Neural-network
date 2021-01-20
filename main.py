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
# def test_loss_function():
#     predicted = Tensor
#     actual = Tensor
#     a = MeanSquareError()
#     predicted = [1, 1, 1, 1, 1, 1]
#     actual = [1, 1, 2, 1, 3, 1]
#     mse = a.loss(predicted, actual)
#     print('loss:', mse)
#     print(predicted[0])
#     print(actual[0])
#     return mse

#test_loss_function()

    

