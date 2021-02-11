"""_______Optimisation of the neural network_____"""

'''
Two functions to use :
-> test_nbr_neuron(list_test) test the impact of the number of neuron in a single hidden layer on the evolution of the error according to the epoch for the training set
-> test_nbr_layer(list_test, n_neuron) test the impact of the number of hidden layer with the same number of neuron on the evolution of the error according to the epoch for the training set
'''
## importations

import Librairie_v2 as lib2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.close()

import random as rd
rd.seed(1)

## Parameters' choice
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
Data_train = pd.read_csv('data_train.csv')
param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

Data_test = pd.read_csv('data_test.csv')
data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])


## User's function


def test_nbr_neuron(list_test):
    color_list=['r','g','b','k','m','c','y']
    color_list *= 3
    k=0
    for i in list_test :
        my_layer1 = lib2.Linear(6,i)
        my_layer2 = lib2.Tanh()
        my_layer3 = lib2.Linear(i,1)
        my_layer4 = lib2.Sigmoid()
        my_NN = lib2.NeuralNet([my_layer1, my_layer2, my_layer3, my_layer4])
        
        
        chi2_list, error_list = lib2.train(my_NN, data_train_input, data_train_target, size_training= train_size, num_epochs = num_epoch_max, lr=my_lr, batch_size=my_batch_size)
        
        data_test_prediction = lib2.prediction(my_NN,data_test_input)
        error_final = lib2.error_round(data_test_prediction, data_test_target)

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
    
    my_layerini1 = lib2.Linear(6,n_neuron)
    my_layerini2 = lib2.Tanh()
    my_layerfini1 = lib2.Linear(n_neuron,1)
    my_layerfini2 = lib2.Sigmoid()
        
    for i in list_test :
        layers_new = [my_layerini1, my_layerini2]
        for j in range(i) :
            layers_new += [lib2.Linear(n_neuron,n_neuron),lib2.Tanh()]
        layers_new += [my_layerfini1, my_layerfini2]
        my_NN = lib2.NeuralNet(layers_new)
        
        
        chi2_list, error_list = lib2.train(my_NN, data_train_input, data_train_target, size_training= train_size, num_epochs = num_epoch_max, lr=my_lr, batch_size=my_batch_size)
        data_test_prediction = lib2.prediction(my_NN,data_test_input)
        
        error_final = lib2.error_round(data_test_prediction, data_test_target)
        
        plt.plot(range(num_epoch_max), error_list, label= str(i),c=color_list[k])
        plt.plot([num_epoch_max],[error_final], marker='o', c=color_list[k])
        plt.xlabel('epoch')
        plt.ylabel('round error')
        
        k+=1
    plt.legend(title='hidden layers')
    plt.show()
























