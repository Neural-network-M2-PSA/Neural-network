""" Librairie V2"""

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

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError
    
    def type(self)-> str:
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
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size,1)



    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (batch_size, input_size)
        """
        self.inputs = inputs
        batch_size = self.inputs.shape[0]
        return np.dot(self.inputs,self.W) + np.dot(np.array([np.ones(batch_size)]).T,self.b.T) #AV formule


    def backward(self, grad: Tensor) -> Tensor:  #a completer
        """
        grad shape is (batch_size, output_size)
        """
        # Compute here the gradient parameters for the layer
        self.grad_w = np.dot(self.inputs.T,grad)
        self.grad_b = np.matrix(grad.mean(axis=0)).T
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

    def backward(self, grad: Tensor) -> Tensor:
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

class SGD(Optimizer):
    def __init__(self) -> None:
        super().__init__()


    def step(self, net: NeuralNet) -> None:
        for layer in net.layers :
            if layer.type() == 'linear':
                layer.W -= self.lr * layer.grad_w
                layer.b -= self.lr * layer.grad_b



##
'''
Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError

        
class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)

'''

##

def train(net: NeuralNet, inputs: Tensor, targets: Tensor,loss: Loss = MeanSquareError(), optimizer: Optimizer = SGD(), num_epochs: int = 5000, size_training : int =100, lr : float = 0.01, batch_size : int = 32) -> None:
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n=0
        
        for i in range(0,size_training,batch_size):
            n+=1
            
            # 1) feed forward
            y_actual = net.forward(inputs[i:i+batch_size])
            
            #2) compute the loss and the gradients
            epoch_loss += loss.loss(targets[i:i+batch_size],y_actual)
            grad_ini = loss.grad(targets[i:i+batch_size],y_actual)
            
            #3)feed backwards
            '''grad_fini n'est pas utile mais dans net.backward() 
            il y a evolution des grad_w et grad_b de chaque layer 
            que l'on utilise ensuite dans l'optimisation  '''
            grad_fini = net.backward(grad_ini)
            
            # 4) update the net 
            optimizer.lr = lr
            optimizer.step(net) #pb esthetique lr attribut de SGD(Optimizer)
        
        epoch_loss = epoch_loss/n #moyenne sur batch
        # Print status every 50 iterations
        if epoch % 50 == 0:
            print(epoch, epoch_loss)



def prediction(net: NeuralNet, inputs: Tensor) -> None:
    return net.forward(inputs)
    

























