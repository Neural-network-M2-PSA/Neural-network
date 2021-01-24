""" Librairie"""

import numpy as np
from numpy import ndarray as Tensor

from typing import (Dict, Tuple, Callable, Sequence, Iterator, NamedTuple)
Func = Callable[[Tensor], Tensor] #comprends pas


## Classe Loss
class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError #ca veut dire quoi ca ?

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    

class MeanSquareError(Loss): #heritage
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        #ce sont des tensor, on peut utiliser les proprietes de numpy.ndarray
        return Tensor.sum((predicted-actual)**2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted-actual) #gradient par rapport a quoi ? AV
        

## Classe Layer

class Layer:
    def __init__(self) -> None:
        #self.params: Dict[str, Tensor] = {}
        #self.grads: Dict[str, Tensor] = {}
        # en se passant de dictionnaire
        return None


    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError




class Linear(Layer): #peut-etre autre chose que lineaire ?
    """
    Inputs are of size (batch_size, input_size) 
    Outputs are of size (batch_size, output_size)
    """


    def __init__(self, input_size: int, output_size: int) -> None:
        '''Inherit from base class Layer'''
        super().__init__() #a quoi ca sert ?
        '''Initialize the weights and bias with random values'''
        #self.params["w"] = np.random.randn(input_size, output_size)
        #self.params["b"] = np.random.randn(output_size)
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)



    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (batch_size, input_size)
        """
        self.inputs = inputs #on peut definir des attributs partout ?
        # Compute here the feed forward pass
        return np.dot(self.W.transpose(),self.inputs) + self.b


    def backward(self, grad: Tensor) -> Tensor:  #a completer
        """
        grad shape is (batch_size, output_size)
        """
        # Compute here the gradient parameters for the layer
        self.grad_w = ...
        self.grads_b = ...  
        # Compute here the feed backward pass
        return ...   





class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: Func, f_prime: Func) -> None: #Comment marche Func ?
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
        return self.f_prime(self.inputs) * grad #grad = ????
        

def tanh(x: Tensor) -> Tensor:
    # Write here the tanh function
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    # Write here the derivative of the tanh
    return 1-(tanh(x))**2 #utilisation fct ci dessus 

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def sigmoid(x: Tensor) -> Tensor:
    # Write here the sigmoid function
    return 1/(1+np.exp(x))

def sigmoid_prime(x: Tensor) -> Tensor:
    # Write here the derivative of the sigmoid
    return sigmoid(x)*(1-sigmoid(x))

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


##

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """
        The forward pass takes the layers in order
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        """
        The backward pass is the other way around
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

































