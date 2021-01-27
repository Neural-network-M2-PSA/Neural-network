import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor

from Loss import MeanSquareError, Loss
from Layer import Layer, Linear, Activation, Sigmoid
from NeuralNet import NeuralNet
from Optimizer import Optimizer, StochasticGradientDescent
from BatchIterator import BatchIterator, DataIterator

Func = Callable[[Tensor], Tensor]

def train(net: NeuralNet, inputs: Tensor, targets: Tensor,
          loss: Loss = MeanSquareError(),
          optimizer: Optimizer = StochasticGradientDescent(),
          iterator: DataIterator = BatchIterator(),
          num_epochs: int = 5000) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            batch.forward(inputs)



        # Print status every 50 iterations
        if epoch % 50 == 0:
            print(epoch, epoch_loss)
