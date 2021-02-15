""" Librairie V2"""
'''
supression classe batch iterator
calcul en batch dans la fct train
ajout fct activation relu et round mais posent pb 
'''

import numpy as np
from numpy import ndarray as Tensor

from typing import (Dict, Tuple, Callable, Sequence, Iterator, NamedTuple)
#Func = Callable[[Tensor], Tensor] #definition du type de fct qu'on utilise a un moment -> a simplifier

import matplotlib.pyplot as plt


## Classe Loss
class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    

class MeanSquareError(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean((predicted-actual)**2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted-actual)/predicted.shape[0]


class ModifiedMeanSquareError(Loss):
    ''' Classe modifiée pour penaliser le minimum local pour l'erreur à 0.25 ( prediction 0.5) à l'aide d'une gaussienne centree en 0.25
'''
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean(1+gauss((predicted-actual)**2))
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        MSE = (predicted-actual)**2
        MSE_prime = 2*(predicted-actual)/predicted.shape[0]
        return MSE_prime * (1 + gauss(MSE)*( 1 - MSE*(MSE - 0.25)/ 0.01**2 ))



## Classe Layer


class Layer:

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
        '''Inherit from base class Layer'''
        super().__init__()
        '''Initialize the weights and bias with random values'''
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros((output_size,1))
        self.type = 'linear'



    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (batch_size, input_size)
        """
        self.inputs = inputs
        batch_size = self.inputs.shape[0]
        return np.dot(self.inputs,self.W) + np.dot(np.array([np.ones(batch_size)]).T,self.b.T)


    def backward(self, grad: Tensor) -> Tensor:
        """
        grad shape is (batch_size, output_size)
        """
        self.grad_w = np.dot(self.inputs.T,grad)
        self.grad_b = np.matrix(grad.mean(axis=0)).T
        return np.dot(grad, self.W.T)



class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f, f_prime) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
        self.type = 'activation'

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad



def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    return 1-(tanh(x))**2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def sigmoid(x: Tensor) -> Tensor:
    return 1/(1+np.exp(x))

def sigmoid_prime(x: Tensor) -> Tensor:
    return sigmoid(x)*(1-sigmoid(x))

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)



##

class NeuralNet:
    def __init__(self, layers: list) -> None:
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
        for layer in self.layers[::-1] :
            grad = layer.backward(grad)
        return grad



##

class Optimizer:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError
    
    def moyenne(self, net: NeuralNet, nbr_batch : int) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self) -> None:
        super().__init__()


    def step(self, net: NeuralNet) -> None:
        for layer in net.layers :
            if layer.type == 'linear':
                layer.W -= self.lr * layer.grad_w
                layer.b -= self.lr * layer.grad_b


##

def train(net: NeuralNet, inputs: Tensor, targets: Tensor,loss: Loss = MeanSquareError(), optimizer: Optimizer = SGD(), num_epochs: int = 5000, size_training : int =100, lr : float = 0.01, batch_size : int = 32) -> list:
    chi2_list = [] ; round_error_list=[]
    for epoch in range(num_epochs):
        chi2_loss = 0.0
        round_error_loss = 0.0
        nbr_batch=0
        
        for i in range(0,size_training,batch_size):
            nbr_batch+=1
            
            # 1) feed forward
            y_actual = net.forward(inputs[i:i+batch_size])
            
            # 2) compute the loss and the gradients
            chi2_loss += loss.loss(targets[i:i+batch_size],y_actual)
            round_error_loss += error_round(targets[i:i+batch_size],y_actual)
            grad_ini = loss.grad(targets[i:i+batch_size],y_actual)
            
            # 3)feed backwards
            '''grad_fini n'est pas utile mais dans net.backward() 
            il y a evolution des grad_w et grad_b de chaque layer 
            que l'on utilise ensuite dans l'optimisation  '''
            grad_fini = net.backward(grad_ini)
            
            # 4) update the net 
            optimizer.lr = lr
            optimizer.step(net) #pb esthetique lr attribut de SGD(Optimizer)
        
        chi2_loss = chi2_loss/nbr_batch
        round_error_loss = round_error_loss/nbr_batch
        chi2_list.append(chi2_loss) 
        round_error_list.append(round_error_loss) 
        
        # Print status every 50 iterations
        if epoch % 50 == 0:
            print('epoch : '+str(epoch)+"/"+str(num_epochs)+", training chi2 error : "+str(chi2_loss)+"\r", end="")
    print('epoch : '+str(epoch)+"/"+str(num_epochs)+", training final chi2 error : "+str(chi2_loss)+'\n')
    
    return chi2_list, round_error_list



def prediction(net: NeuralNet, inputs: Tensor) -> None:
    return net.forward(inputs)




    























