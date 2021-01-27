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


class Linear(Layer):
    """
    Inputs are of size (batch_size, input_size)
    Outputs are of size (batch_size, output_size)
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # Inherit from base class Layer
        super().__init__()
        # Initialize the weights and bias with random values
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (batch_size, input_size)
        """
        self.inputs = inputs
        z = Tensor
        # Compute here the feed forward pass

        ##Pour le linear layer (OK)
        #print("w: ",self.params["w"], " Shape: ",self.params["w"].shape)
        print("Shape 1: ",np.dot(np.transpose(self.params["w"]), self.inputs).shape)
        print("Shape 2: ",self.params["b"].shape)
        batch_size = self.inputs.shape[0]
        z = np.dot(np.transpose(self.params["w"]), self.inputs) + np.dot(np.array([np.ones(batch_size)]).T,self.param["b"].T)
        print('z = ', z)
        return z

    def backward(self, grad: Tensor) -> Tensor:
         """
         grad shape is (batch_size, output_size)
         """
         res = Tensor
         b = MeanSquareError()
         w = MeanSquareError()
         # Compute here the gradient parameters for the layer
         self.grads["w"] = np.dot(grad, self.inputs)
         self.grads["b"] = b.grad(self.params["b"], grad)
         # Compute here the feed backward pass
         res = np.dot(self.grads["w"],self.grads["b"])
         #print("res : ", res)
         #print("Backward effectué")
         return res

    def type(self) -> str:
        return 'linear'

class Activation(Layer):
    """
    An activation layer just applies a function
    element wise to its inputs
    """
    def __init__(self, f: Func, f_prime: Func) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad

    def type(self)->str:
        return 'Activation'

def tanh(x: Tensor) -> Tensor:
    res = Tensor
    res = np.tanh(x)
    return res


def tanh_prime(x: Tensor) -> Tensor:
    res = Tensor
    res = 1 - np.tanh(x)**2
    return res


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

def sigmoid(x: Tensor) -> Tensor:
    res = Tensor
    res = 0.5 + 0.5*np.tanh(x/2)
    return res


def sigmoid_prime(x: Tensor) -> Tensor:
    res = Tensor
    #res =1-0.25*np.tanh(x/2)**2)
    return res

#def relu(x: Tensor) -> Tensor:
    #res = Tensor


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)
