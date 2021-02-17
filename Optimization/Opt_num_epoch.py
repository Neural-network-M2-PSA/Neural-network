"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______Opt_num_epoch_____"""

'''
This file contains two functions:
"Opt_nbr_epoch" returns a graph with the evolution of the error for the training and testing set. It also returns in a
 DataFrame the mean squared error et our round error of the training and testing set according to the number of epoch.
 This DataFrame is also save as a csv file.

"Opt_learning_rate" returns a graph in order to compare the evolution of the training set's error according to various 
learning rate (given by the user in the list list_learning_rate).

'''
# Libraries
import numpy as np
from numpy import ndarray as Tensor
import pandas as pd
import matplotlib.pyplot as plt

# Imports
import Neural_Network_Library.loss as Loss
import Neural_Network_Library.layer as Layer
import Neural_Network_Library.error_round as Error_round
import Neural_Network_Library.activation_functions as ActivationFunctions
import Neural_Network_Library.neural_network as Neural_network
import Neural_Network_Library.optimizer as OptimizerClass
import Neural_Network_Library.user as User


plt.close()

# Parameters' choice

'''seed '''
np.random.seed(1)

'''Construction of the neural network '''
my_layer1 = Layer.Linear(6,4)
my_layer2 = ActivationFunctions.Tanh()
my_layer5 = Layer.Linear(4,2)
my_layer6 = ActivationFunctions.Tanh()
my_layer3 = Layer.Linear(2,1)
my_layer4 = ActivationFunctions.Sigmoid()
my_NN = Neural_network.NeuralNet([my_layer1, my_layer2, my_layer5, my_layer6, my_layer3, my_layer4])

'''Maximal number of epochs '''
Nmax = 50000

'''size of the training set and the testing set '''
train_size = 3000
test_size = 1500

'''size of the batch'''
my_batch_size=100

''' learning rate'''
my_lr=0.0005

'''importation of the training and testing data'''
Data_train = pd.read_csv('D:\data_train.csv')
Data_test = pd.read_csv('D:\data_test.csv')

param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])


# Modified training function

def train_prediction(net: Neural_network.NeuralNet, inputs_train: Tensor, targets_train: Tensor, inputs_test: Tensor, targets_test: Tensor, loss: Loss.Loss = Loss.MeanSquareError(), optimizer: OptimizerClass.Optimizer = OptimizerClass.SGD(), num_epochs: int = 5000, batch_size : int = 32) -> None:
    Data = pd.DataFrame(columns = ('MSE_train', 'MSE_test', 'error_round_train', 'error_round_test'))
    size_training = inputs_train.shape[0]
    for epoch in range(num_epochs):
        Chi2_train = 0.0
        error_round_train = 0.0
        nbr_batch=0
        
        for i in range(0,size_training,batch_size):
            nbr_batch+=1
            
            # 1) feed forward
            y_actual = net.forward(inputs_train[i:i+batch_size])
            
            # 2) compute the loss and the gradients
            Chi2_train += loss.loss(targets_train[i:i+batch_size],y_actual)
            grad_ini = loss.grad(targets_train[i:i+batch_size],y_actual)
            
            # 3)feed backwards
            grad_fini = net.backward(grad_ini)
            
            # 4) update the net 
            optimizer.step(net, n_epoch = epoch)
            
            error_round_train += Error_round.error_round(targets_train[i:i+batch_size], y_actual)
        
        Chi2_train = Chi2_train/nbr_batch
        error_round_train = error_round_train /nbr_batch
        
        y_actual_test = net.forward(inputs_test)
        Chi2_test = loss.loss(targets_test, y_actual_test)
        error_round_test = Error_round.error_round(targets_test, y_actual_test)

        
        if epoch % 100 == 0:
            print('epoch : '+str(epoch)+"/"+str(num_epochs)+"\r", end="")
    
        datanew = pd.DataFrame({'Chi2_train':[Chi2_train], 'Chi2_test':[Chi2_test], 'error_round_train':[error_round_train], 'error_round_test':[error_round_test]})
        Data = Data.append(datanew)
    
    Data.to_csv('Opt_num_epoch_backup.csv',index=False)
    
    return Data
    



# User's function


def Opt_nbr_epoch() :
    '''
    Evolution of the chi2 et our special round error of the training and testing set according to the number of epoch.
    '''
    Data = train_prediction(my_NN, data_train_input, data_train_target, data_test_input, data_test_target, num_epochs = Nmax,optimizer = Optimizer.SGD(lr = my_lr) batch_size = my_batch_size)
    print(Data)
    plt.plot(range(Nmax), Data['MSE_train'], label='training')
    plt.plot(range(Nmax), Data['MSE_test'], label='testing')
    
    plt.xlabel('epoch')
    plt.ylabel(r'Mean Squared Error')
    plt.legend()
    
    plt.show()




def Opt_learning_rate(list_learning_rate):
    for my_lr in list_learning_rate :
        Data = User.train(my_NN, data_train_input, data_train_target, num_epochs = Nmax, optimizer = Optimizer.SGD(lr = my_lr), batch_size = my_batch_size)[1]
        plt.plot(range(Nmax), Data, label=str(my_lr))
    
    plt.xlabel('epoch')
    plt.ylabel(' training mean squared error')
    plt.legend(title='learning rate impact')
    plt.show()












