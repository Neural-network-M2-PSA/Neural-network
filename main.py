import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor
from Loss import Loss, MeanSquareError
Func = Callable[[Tensor], Tensor]
from Layer import Layer, Linear, Activation
from Network import Network

import pandas as pd

#DATA = pd.read_csv('D:\dataset.csv', delimiter=',')


def tanh(x: Tensor) -> Tensor:
    res = Tensor
    res = np.tanh(x)
    return res


def tanh_prime(x: Tensor) -> Tensor:
    res = Tensor
    res = 1 - np.tanh(x)**2
    return res

def loss(self, predicted: Tensor, actual: Tensor) -> float:

    mse = 0
    for i in range(len(predicted)):
        mse += ((predicted[i] - actual[i])**2)
        print('predicted[i]: ', predicted[i],'+actual[i]: ',actual[i], 'difference au carré: ',(predicted[i] - actual[i])**2)

    return mse

def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
    grad=0
    for j in range(len(predicted)):
        grad+= 2*(predicted[j]-actual[j])
        print('predicted[i]: ', predicted[j], '+actual[i]: ', actual[j], 'grad: ', predicted[j] - actual[j])
    return grad


# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(Linear(2, 3))
net.add(Activation(tanh, tanh_prime))
net.add(Linear(3, 1))
net.add(Activation(tanh, tanh_prime))

# train
net.use(loss, grad)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)








############################################################
## TEST PRELIMINAIRES
#Test pour voir si la loss function fonctionne
# def test_loss_function():
#     predicted = Tensor
#     actual = Tensor
#     a = MeanSquareError()
#     b = MeanSquareError()
#     predicted = [1, 1, 1, 1, 1, 1]
#     actual = [1, 1, 2, 1, 3, 1]

    #Test pour la loss function et le grad de la loss function
    # loss = a.loss(predicted, actual)
    # grad = b.grad(predicted, actual)
    #print('loss:', loss)
    #print('grad:', grad)

    #Pour un seul neurone
    #layer_test = Linear(6,1)
    #layer_test.forward(predicted)


    #return 0



# # Test pour voir si un seul neurone fonctionne
# def test_one_neuron():
#     predicted = Tensor
#     predicted = [1, 1, 1, 1, 1, 1]
#
#     # Pour un seul neurone
#     layer_test = Linear(6,1)
#     layer_test.forward(predicted)
#
#     return 0
#
# #test_one_neuron()
#
# def test_linear_layer():
#     predicted = Tensor
#     predicted = [1, 1, 1, 1]
#     #print("Predicted shape: ", predicted.shape)
#
#
#     layer_test = Linear(4,4)
#     layer_test.forward(predicted)
#     layer_test.backward(predicted)
#
#     return 0
# test_linear_layer()
######################################################################