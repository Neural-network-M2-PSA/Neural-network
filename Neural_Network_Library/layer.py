"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______layer_____"""
'''
This file contains the class Layer, and the daughter classes Linear and Activation which inherit of the base class Layer. 

In the class Linear, the inputs have size (batch_size, input_size) and the outputs have size 
(batch_size, output_size).
In the function forward, inputs shape is (batch_size, input_size).
In the function backward, grad shape is (batch_size, output_size).

In the class Activation, the layer applies an activation function to its inputs.

These two classes have different argument for self.type: 'linear' or 'activation'


'''


# Libraries
import numpy as np
from numpy import ndarray as Tensor

class Layer:

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):

    def __init__(self, input_size: int, output_size: int) -> None:

        super().__init__()
        '''Initialize the weights and bias with random values'''
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros((output_size, 1))
        self.type = 'linear'

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        batch_size = self.inputs.shape[0]
        return np.dot(self.inputs, self.W) + np.dot(np.array([np.ones(batch_size)]).T, self.b.T)

    def backward(self, grad: Tensor) -> Tensor:
        self.grad_w = np.dot(self.inputs.T, grad)
        self.grad_b = np.matrix(grad.mean(axis=0)).T
        return np.dot(grad, self.W.T)


class Activation(Layer):

    def __init__(self, f, f_prime) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
        self.type = 'activation'

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad

