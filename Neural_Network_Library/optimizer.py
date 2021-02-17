"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______optimizer_____"""
'''
This file contains the class Optimizer, and the class SGD which inherit of the base class Optimizer.
SGD means Stochastic gradient descent.  



'''
# Imports
import Neural_Network_Library.neural_network as Neural_network

class Optimizer:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: Neural_network.NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self) -> None:
        super().__init__()

    def step(self, net: Neural_network.NeuralNet) -> None:
        for layer in net.layers:
            if layer.type == 'linear':
                layer.W -= self.lr * layer.grad_w
                layer.b -= self.lr * layer.grad_b

