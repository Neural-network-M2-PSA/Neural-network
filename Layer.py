import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor

from Loss import MeanSquareError, Loss

Func = Callable[[Tensor], Tensor]

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

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
        # Inherit from base class Layer
        super().__init__()
        # Initialize the weights and bias with random values
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (batch_size, input_size)
        """
        self.inputs = inputs
        # Compute here the feed forward pass
        #print(self.params["w"])
        #print(self.params["b"])

        #Pour un seul neurone
        #z = np.dot(np.transpose(self.params["w"]), np.transpose(self.inputs)) + self.params["b"]
        #print('input_transpose shape: ', np.transpose(self.inputs).shape)
        #print('w_transpose shape: ', np.transpose(self.params["w"]).shape)
        #print('z = ',z)

        ##Pour le linear layer (FAUX)
        # print(self.params["w"])
        # print('w_transpose shape: ', np.transpose(self.params["w"]).shape)
        # print(np.transpose(self.inputs))
        # print('input_transpose shape: ', np.transpose(self.inputs).shape)
        # print(np.transpose(self.params["b"]))
        # print('b_ shape: ', np.transpose(self.params["b"]).shape)
        # z = np.dot(np.transpose(self.params["w"]), np.transpose(self.inputs)) + self.params["b"]
        # print('z = ', z)
        # print('z_ shape: ', z.shape)

        ##Pour le linear layer (OK)
        print(self.params["w"])
        print(self.inputs)
        print(self.params["b"])
        z = np.dot(self.params["w"], self.inputs) + self.params["b"]
        print('z = ', z)

        return z

    def backward(self, grad: Tensor) -> Tensor:
         """
         grad shape is (batch_size, output_size)
         """
         a = Tensor
         b = MeanSquareError()
         w = MeanSquareError()
         # Compute here the gradient parameters for the layer
         self.grads["w"] = w.grad(self.params["w"], self.inputs)
         self.grads["b"] = w.grad(self.params["b"], self.inputs)
         # Compute here the feed backward pass
         a = [self.grads["w"], self.grads["b"]]
         print("a : ", a)
         return a

