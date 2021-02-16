"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______neural_network_____"""
'''
This file contains the class NeuralNet.

In this class, the function forward takes the layers in order and the function backward is the other way around.(???)


'''
#Library
from numpy import ndarray as Tensor

class NeuralNet:
    def __init__(self, layers: list) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


    def backward(self, grad: Tensor) -> Tensor:
        """
        The backward pass is the other way around
        """
        for layer in self.layers[::-1] :
            grad = layer.backward(grad)
        return grad