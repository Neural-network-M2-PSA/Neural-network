"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______user_____"""
'''
This file contains two functions: train and predict. 
The function "train" trains the neural network. Its takes in arguments lots of elements: the network, the inputs, the
function of loss and the optimizer. There are other paramaters which are defined by default but the user can modifie 
them: the number of epochs, the size of the training, the learning rate lr and the batch size. This functions has 4 
mains steps : 1) feed the forward, 2) compute the loss and the gradient, 3) feed the backward and 4) update the neural 
network. 
Remarke: "grad_fini" isn't necessary here. But, there is an evolution of grad_w and grad_b of each layer in 
net.backgrad(), which is useful later for the optimization. 

The function "prediction" predicts takes in arguments the neural network and the input. Then, this function predict
the outputs using the function forward.   


'''

#Library
from numpy import ndarray as Tensor


#Imports
import Neural_Network_Library.error_round as error_round2
import Neural_Network_Library.loss as Loss
import Neural_Network_Library.neural_network as Neural_network
import Neural_Network_Library.optimizer as OptimizerClass


def train(net: Neural_network.NeuralNet, inputs: Tensor, targets: Tensor, loss: Loss = Loss.MeanSquareError(),
          optimizer: OptimizerClass.Optimizer = OptimizerClass.SGD(), num_epochs: int = 5000, size_training: int = 100,
          lr: float = 0.01, batch_size: int = 32) -> list:
    chi2_list = [];
    round_error_list = []
    for epoch in range(num_epochs):
        chi2_loss = 0.0
        round_error_loss = 0.0
        nbr_batch = 0

        for i in range(0, size_training, batch_size):
            nbr_batch += 1

            # 1) Feed forward
            y_actual = net.forward(inputs[i:i + batch_size])

            # 2) Compute the loss and the gradient
            chi2_loss += loss.loss(targets[i:i + batch_size], y_actual)
            round_error_loss += error_round2.error_round(targets[i:i + batch_size], y_actual)
            grad_ini = loss.grad(targets[i:i + batch_size], y_actual)

            # 3) Feed backwards
            grad_fini = net.backward(grad_ini)

            # 4) Update the net
            optimizer.lr = lr
            optimizer.step(net)

        chi2_loss = chi2_loss / nbr_batch
        round_error_loss = round_error_loss / nbr_batch
        chi2_list.append(chi2_loss)
        round_error_list.append(round_error_loss)

        # Print status every 50 iterations
        if epoch % 50 == 0:
            print('epoch : ' + str(epoch) + "/" + str(num_epochs) + ", training chi2 error : " + str(chi2_loss) + "\r",
                  end="")
    print('epoch : ' + str(epoch) + "/" + str(num_epochs) + ", training final chi2 error : " + str(chi2_loss) + '\n')

    return chi2_list, round_error_list


def prediction(net: Neural_network.NeuralNet, inputs: Tensor) -> None:
    return net.forward(inputs)