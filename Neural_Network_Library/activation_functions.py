"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______activation_functions_____"""
'''
Basic activation functions of a neural network: 
-Tanh
-Sigmoid
-Identity
-Relu
'''

# Libraries
import numpy as np
from numpy import ndarray as Tensor

#Imports
import Neural_Network_Library.layer as Layer

'''
____________________________________
Activation function : Tanh
____________________________________
'''

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    return 1-(tanh(x))**2

class Tanh(Layer.Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


'''
____________________________________
Activation function : Sigmoïd
____________________________________
'''

def sigmoid(x: Tensor) -> Tensor:
    return 1/(1+np.exp(x))

def sigmoid_prime(x: Tensor) -> Tensor:
    return sigmoid(x)*(1-sigmoid(x))

class Sigmoid(Layer.Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


'''
____________________________________
Activation function : Identity
____________________________________
'''

def identity(x: Tensor) -> Tensor:
    return x

def identity_prime(x: Tensor) -> Tensor:
    return 1

class Identity(Layer.Activation):
    def __init__(self):
        super().__init__(identity, identity_prime)


'''
____________________________________
Activation function : Relu
____________________________________
'''

def relu(x: Tensor) -> Tensor:
    return x

def relu_prime(x: Tensor) -> Tensor:
    return 1

class Relu(Layer.Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)