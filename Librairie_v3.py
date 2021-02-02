""" Librairie V3"""
'''
Changements :
calculs sans batch en 1D (plus de moyenne sur b)
suppression de la classe optimizer avec le learning rate directement dans backward de Linear(Layer)
'''

import numpy as np
from numpy import ndarray as Tensor

from typing import (Dict, Tuple, Callable, Sequence, Iterator, NamedTuple)
Func = Callable[[Tensor], Tensor] #definition du type de fct qu'on utilise a un moment -> a simplifier


## Classe Loss
class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError #ca veut dire quoi ca ?

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    

class MeanSquareError(Loss): #heritage
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        #ce sont des tensor, on peut utiliser les proprietes de numpy.ndarray
        return np.mean((predicted-actual)**2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted-actual)/predicted.shape[0]

## Classe Layer


class Layer:

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor, lr: float=0.1) -> Tensor:
        raise NotImplementedError
    
    def type(self)-> str:
        raise NotImplementedError



class Linear(Layer): #peut-etre autre chose que lineaire ?
    """
    Inputs are of size (input_size)  array simple ! pas de tensor
    Outputs are of size (output_size)
    """


    def __init__(self, input_size: int, output_size: int) -> None:
        '''Inherit from base class Layer'''
        super().__init__() #a quoi ca sert ?
        '''Initialize the weights and bias with random values'''
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(1,output_size)



    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (input_size)
        """
        self.inputs = inputs
        
        return np.dot(self.inputs,self.W) + self.b


    def backward(self, grad: Tensor, lr: float=0.1) -> Tensor:  #a completer
        """
        grad shape is (output_size)
        """
        # Compute here the gradient parameters for the layer
        self.grad_w = np.dot(self.inputs.T,grad)
        self.grad_b = grad
        
        #optimize
        self.W -= lr * self.grad_w
        self.b -= lr * self.grad_b
        
        return np.dot(grad, self.W.T)

    def type(self)-> str:
        return 'linear'


class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: Func, f_prime: Func) -> None: #Func -> type des fct
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor, lr: float=0.1) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad

    def type(self)-> str:
        return 'activation'



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


'''Fct d'activation test : '''
def arondi(x: Tensor) -> Tensor:
    return np.round(x)

def arondi_prime(x: Tensor) -> Tensor:
    return 1 #ce n'est pas la dériver de round, juste pour tester

class Arondi(Activation):
    def __init__(self):
        super().__init__(arondi, arondi_prime)



def relu(x: Tensor) -> Tensor:
    x[x<0]=0
    return x

def relu_prime(x: Tensor) -> Tensor:
    x[x<0]=0
    x[x>0]=1
    return x

class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)



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


    def backward(self, grad: Tensor, lr: float) -> Tensor:
        """
        The backward pass is the other way around
        """
        for layer in self.layers[::-1] :
            grad = layer.backward(grad,lr)
        return grad



##
'''
class Optimizer:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self) -> None:
        super().__init__()


    def step(self, net: NeuralNet) -> None:
        for layer in net.layers :
            if layer.type() == 'linear':
                layer.W -= self.lr * layer.grad_w
                layer.b -= self.lr * layer.grad_b

'''
##

def train(net: NeuralNet, inputs: Tensor, targets: Tensor,loss: Loss = MeanSquareError(), num_epochs: int = 5000, size_training : int =100, lr : float = 0.01) -> None:
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for i in range(size_training):
            
            # 1) feed forward
            y_actual = net.forward(np.array([inputs[i]])) #trouver formule + jolie
            
            # 2) compute the loss and the gradients
            epoch_loss += loss.loss(targets[i],y_actual)
            grad_ini = loss.grad(targets[i],y_actual)
            
            # 3)feed backwards
            '''grad_fini n'est pas utile mais dans net.backward() 
            il y a evolution des grad_w et grad_b de chaque layer 
            que l'on utilise ensuite dans l'optimisation  '''
            grad_fini = net.backward(grad_ini, lr)
            
            '''
            # 4) update the net 
            optimizer.lr = lr
            optimizer.step(net) #pb esthetique lr attribut de SGD(Optimizer)
            '''
        epoch_loss = epoch_loss/size_training #moyenne sur batch
        # Print status every 50 iterations
        if epoch % 50 == 0:
            print(epoch, epoch_loss)



def prediction(net: NeuralNet, inputs: Tensor) -> None:
    return net.forward(inputs)
    

























