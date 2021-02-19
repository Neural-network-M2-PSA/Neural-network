"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______optimizer_____"""
'''
This file contains the class Optimizer, and the classes SGD and DecaySGD which inherit of the base class Optimizer.
SGD means Stochastic Gradient Descent with a constant learning rate which is personalizable.
DecaySGD is also a Stochastic Gradient Descent but with an adaptive learning rate : a decay of parameters lr_initial and decay_coeff.
'''

#Library
import numpy as np

# Imports
import Neural_Network_Library.neural_network as Neural_network

class Optimizer:
    def step(self, net: Neural_network.NeuralNet,n_epoch: int = 1) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def step(self, net: Neural_network.NeuralNet, n_epoch: int = 1) -> None:
        for layer in net.layers :
            if layer.type == 'linear':
                layer.W -= self.lr * layer.grad_w
                layer.b -= self.lr * layer.grad_b


class DecaySGD(Optimizer):
    def __init__(self, initial_lr: float = 0.1, decay_coeff: float = 1/200 ) -> None:
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_coeff = decay_coeff

    def step(self, net: Neural_network.NeuralNet, n_epoch: int = 1) -> None:
        for layer in net.layers :
            if layer.type == 'linear':
                layer.W -= self.initial_lr * np.exp(-self.decay_coeff *n_epoch)* layer.grad_w
                layer.b -= self.initial_lr * np.exp(-self.decay_coeff *n_epoch) * layer.grad_b
