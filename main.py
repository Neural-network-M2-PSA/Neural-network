import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor
from Loss import Loss, MeanSquareError
Func = Callable[[Tensor], Tensor]
from Layer import Layer, Linear

def test_hello():
    print('hello')
#test_hello()

#Test pour voir si la loss function fonctionne
def test_loss_function():
    predicted = Tensor
    actual = Tensor
    a = MeanSquareError()
    b = MeanSquareError()
    predicted = [1, 1, 1, 1, 1, 1]
    actual = [1, 1, 2, 1, 3, 1]

    #Test pour la loss function et le grad de la loss function
    loss = a.loss(predicted, actual)
    grad = b.grad(predicted, actual)
    #print('loss:', loss)
    #print('grad:', grad)

    #Pour un seul neurone
    #layer_test = Linear(6,1)
    #layer_test.forward(predicted)


    return 0



# Test pour voir si un seul neurone fonctionne
def test_one_neuron():
    predicted = Tensor
    predicted = [1, 1, 1, 1, 1, 1]

    # Pour un seul neurone
    layer_test = Linear(6,1)
    layer_test.forward(predicted)

    return 0
    
#test_one_neuron()

def test_linear_layer():
    predicted = Tensor
    predicted = [1, 1, 1, 1]

    # Pour un seul neurone
    layer_test = Linear(4,4)
    layer_test.forward(predicted)
    layer_test.backward(predicted)

    return 0
test_linear_layer()