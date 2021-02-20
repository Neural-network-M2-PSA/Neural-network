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
The function "train" trains the neural network. Its takes in arguments lots of elements: the network, the inputs, the target,
the function of loss (default = MSE() ), the optimizer ( default : SGD(), alternative : DecaySGD()),
the total number of epoch (default 500) and the batch_size (default : 100).

This functions has 4 mains steps : 
1) feed the forward, 2) compute the loss and the gradient, 3) feed the backward and 4) update the neural network. 
Remarke: "grad_fini" isn't necessary here. But, there is an evolution of grad_w and grad_b of each layer in 
net.backgrad(), which is useful later for the optimization. 

The function "prediction" takes in arguments the neural network and the input. Then, this function predicts
the outputs using the function forward. This function must be used after a training of a neural network.


'''

# Library
from numpy import ndarray as Tensor


#Imports
import error_round as Error_round
import loss as Loss
import neural_network as Neural_network
import optimizer as OptimizerClass


def train(net: Neural_network.NeuralNet, inputs: Tensor, targets: Tensor, loss: Loss = Loss.MeanSquareError(),
          optimizer: OptimizerClass.Optimizer = OptimizerClass.SGD(), num_epochs: int = 5000, batch_size: int = 32) -> tuple:
    chi2_list = []; round_error_list = []
    size_training = inputs.shape[0] 
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
            round_error_loss += Error_round.error_round(targets[i:i + batch_size], y_actual)
            grad_ini = loss.grad(targets[i:i + batch_size], y_actual)

            # 3) Feed backwards
            grad_fini = net.backward(grad_ini)

            # 4) Update the net
            optimizer.step(net, n_epoch = epoch)

        chi2_loss = chi2_loss / nbr_batch
        round_error_loss = round_error_loss / nbr_batch
        chi2_list.append(chi2_loss)
        round_error_list.append(round_error_loss)

        # Print status every 50 iterations
        if epoch % 50 == 0:
            print('\r epoch : ' + str(epoch) + "/" + str(num_epochs) + ", training mean squared error : " + str(chi2_loss) + "\r",
                  end="")
    print('epoch : ' + str(epoch) + "/" + str(num_epochs) + ", training final mean squared error : " + str(chi2_loss) + '\n')

    return chi2_list, round_error_list


def prediction(net: Neural_network.NeuralNet, inputs: Tensor) -> Tensor:
    return net.forward(inputs)
