import numpy as np
from numpy import ndarray as Tensor

def loss(predicted: Tensor, actual: Tensor) -> float:
    mse = 0
    for i in range(len(predicted)):
        mse += ((predicted[i] - actual[i]) ** 2)
        print('predicted[i]: ', predicted[i], '+actual[i]: ', actual[i], 'difference au carré: ',
              (predicted[i] - actual[i]) ** 2)

    return mse


def grad(predicted: Tensor, actual: Tensor) -> Tensor:
    grad = 0
    for j in range(len(predicted)):
        grad += 2 * (predicted[j] - actual[j])
        print('predicted[i]: ', predicted[j], '+actual[i]: ', actual[j], 'grad: ', predicted[j] - actual[j])
    return grad