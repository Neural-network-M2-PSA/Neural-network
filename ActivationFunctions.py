import numpy as np
from numpy import ndarray as Tensor

##################################################
##FONCTION D'ACTIVATION TANH
##################################################

def tanh(x: Tensor) -> Tensor:
    res = Tensor
    res = np.tanh(x)
    return res


def tanh_prime(x: Tensor) -> Tensor:
    res = Tensor
    res = 1 - np.tanh(x)**2
    return res

##################################################
##FONCTION D'ACTIVATION RELU
##################################################

def relu(x: Tensor) -> Tensor:
    res = Tensor
    if x.all() <= 0:
        res = 0
    if x.all() > 0:
        res= x
    return res


def relu_prime(x: Tensor) -> Tensor:
    res = Tensor
    if x.all() <=0:
        res = 0
    if x.all() > 0:
        res = 1
    return res

##################################################
##FONCTION D'ACTIVATION  SIGMOID
##################################################

def sigmoid(x: Tensor) -> Tensor:
    res = Tensor
    res = 0.5 + 0.5*np.tanh(x/2)
    return res


def sigmoid_prime(x: Tensor) -> Tensor:
    res = Tensor
    res =1-0.25*np.tanh(x/2)**2
    return res



