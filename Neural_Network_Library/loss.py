"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______loss_____"""
'''
This file contains the class Loss, and the classes MeanSquareError and ModifiedMeanSquareError which inherit of the base
class Loss. 

In the class MeanSquareError, we compute the loss. For that, we measures the average of the squares of the errors, that 
is, the average squared difference between the estimated values and the actual value. MSE is a risk function, 
corresponding to the expected value of the squared error loss. We also compute the grad by derivation of the loss. 

In the class ModifiedMeanSquareError, we modified the class MeanSquareError to.....


'''


#Libraries
import numpy as np
from numpy import ndarray as Tensor

#Imports
import Neural_Network_Library.gauss as Gauss

class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MeanSquareError(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual) / predicted.shape[0]


class ModifiedMeanSquareError(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean(1 + Gauss.gauss((predicted - actual) ** 2))

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        MSE = (predicted - actual) ** 2
        MSE_prime = 2 * (predicted - actual) / predicted.shape[0]
        return MSE_prime * (1 + Gauss.gauss(MSE) * (1 - MSE * (MSE - 0.25) / 0.01 ** 2))