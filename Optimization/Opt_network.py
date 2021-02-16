"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______Opt_network_____"""

'''
This file contains two functions:
"test_nbr_neuron" tests the impact of the number of neurons in a single hidden layer on the evolution of the error
 according to the epoch for the training set
"test_nbr_layer" tests the impact of the number of hidden layers with the same number of neurons on the evolution 
of the error according to the epoch for the training set
'''
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports


import Neural_Network_Library.layer as Layer
import Neural_Network_Library.error_round as Error_round
import Neural_Network_Library.activation_functions as ActivationFunctions
import Neural_Network_Library.neural_network as Neural_network
import Neural_Network_Library.user as User

plt.close()
np.random.seed(1)

# Parameters' choice
'''Maximal number of epochs '''
num_epoch_max = 2000

'''size of the training set and the testing set '''
train_size = 3000
test_size = 1500

'''size of the batch'''
my_batch_size=100

''' learning rate'''
my_lr=0.001

'''importation of the training and testing data'''
Data_train = pd.read_csv('D:\data_train.csv')
param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

Data_test = pd.read_csv('D:\data_test.csv')
data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])


# User's function

def test_nbr_neuron(list_test):
    color_list=['r','g','b','k','m','c','y']
    color_list *= 3
    k=0
    for i in list_test :
        my_layer1 = Layer.Linear(6,i)
        my_layer2 = ActivationFunctions.Tanh()
        my_layer3 = Layer.Linear(i,1)
        my_layer4 = ActivationFunctions.Sigmoid()
        my_NN = Neural_network.NeuralNet([my_layer1, my_layer2, my_layer3, my_layer4])
        
        
        chi2_list, error_list = User.train(my_NN, data_train_input, data_train_target, size_training= train_size, num_epochs = num_epoch_max, lr=my_lr, batch_size=my_batch_size)
        
        data_test_prediction = User.prediction(my_NN,data_test_input)
        error_final = Error_round.error_round(data_test_prediction, data_test_target)

        plt.plot(range(num_epoch_max), error_list, label= str(i), c=color_list[k])
        plt.plot([num_epoch_max],[error_final], marker='o', c=color_list[k])
        plt.xlabel('epoch')
        plt.ylabel('training round error')
        
        k+=1
    plt.legend(title='neurons')
    plt.show()



def test_nbr_layer(list_test, n_neuron):
    color_list=['r','g','b','k','m','c','y']
    color_list *= 3
    k=0
    
    my_layerini1 = Layer.Linear(6,n_neuron)
    my_layerini2 = ActivationFunctions.Tanh()
    my_layerfini1 = Layer.Linear(n_neuron,1)
    my_layerfini2 = ActivationFunctions.Sigmoid()
        
    for i in list_test :
        layers_new = [my_layerini1, my_layerini2]
        for j in range(i) :
            layers_new += [Layer.Linear(n_neuron,n_neuron),Layer.Tanh()]
        layers_new += [my_layerfini1, my_layerfini2]
        my_NN = Neural_network.NeuralNet(layers_new)
        
        
        chi2_list, error_list = User.train(my_NN, data_train_input, data_train_target, size_training= train_size, num_epochs = num_epoch_max, lr=my_lr, batch_size=my_batch_size)
        data_test_prediction = User.prediction(my_NN,data_test_input)
        
        error_final = Error_round.error_round(data_test_prediction, data_test_target)
        
        plt.plot(range(num_epoch_max), error_list, label= str(i),c=color_list[k])
        plt.plot([num_epoch_max],[error_final], marker='o', c=color_list[k])
        plt.xlabel('epoch')
        plt.ylabel('round error')
        
        k+=1
    plt.legend(title='hidden layers')
    plt.show()
























