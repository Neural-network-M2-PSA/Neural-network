"""_______Optimisation of the hyper-parametres_____"""

'''
Two functions to use :
-> Opt_nbr_epoch() returns a graph with the evolution of the error for the training and testing set
-> Opt_learning_rate(list_learning_rate) returns a graph in order to comparate the evolution of the training set's error according to various learning rate (given by the user in the list list_learning_rate.

'''


## importations
import Librairie_v2 as lib2
import numpy as np
from numpy import ndarray as Tensor
import pandas as pd

import matplotlib.pyplot as plt
plt.close()

## Parameters' choice

'''Construction of the neural network '''
my_layer1 = lib2.Linear(6,3)
my_layer2 = lib2.Tanh()
my_layer3 = lib2.Linear(3,1)
my_layer4 = lib2.Sigmoid()
my_NN = lib2.NeuralNet([my_layer1, my_layer2, my_layer3, my_layer4])

'''Maximal number of epochs '''
Nmax = 5000

'''size of the training set and the testing set '''
train_size = 3000
test_size = 1500

'''size of the batch'''
my_batch_size=100

''' learning rate'''
my_lr=0.001

'''importation of the training and testing data'''
Data_train = pd.read_csv('data_train.csv')
Data_test = pd.read_csv('data_test.csv')

param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])


## Modified function

def train_prediction(net: lib2.NeuralNet, inputs_train: Tensor, targets_train: Tensor, inputs_test: Tensor, targets_test: Tensor, loss: lib2.Loss = lib2.MeanSquareError(), optimizer: lib2.Optimizer = lib2.SGD(), num_epochs: int = 5000, size_training : int =100, lr : float = 0.01, batch_size : int = 32) -> None:
    '''
    This return function returns in a DataFrame the chi2 et our special round error of the training and testing set according to the number of epoch.
    This DataFrame is also save as a csv file.
    '''
    
    Data = pd.DataFrame(columns = ('Chi2_train', 'Chi2_test', 'error_round_train', 'error_round_test'))
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
            optimizer.lr = lr
            optimizer.step(net)
            
            error_round_train += lib2.error_round(targets_train[i:i+batch_size], y_actual)
        
        Chi2_train = Chi2_train/nbr_batch
        error_round_train = error_round_train /nbr_batch
        
        y_actual_test = net.forward(inputs_test)
        Chi2_test = loss.loss(targets_test, y_actual_test)
        error_round_test = lib2.error_round(targets_test, y_actual_test)

        
        if epoch % 100 == 0:
            print('epoch : '+str(epoch)+"/"+str(num_epochs)+"\r", end="")
    
        datanew = pd.DataFrame({'Chi2_train':[Chi2_train], 'Chi2_test':[Chi2_test], 'error_round_train':[error_round_train], 'error_round_test':[error_round_test]})
        Data = Data.append(datanew)
    
    Data.to_csv('Opt_num_epoch.csv',index=False)
    
    return Data
    



## User's function


def Opt_nbr_epoch() :
    '''
    Evolution of the chi2 et our special round error of the training and testing set according to the number of epoch.
    '''
    Data = train_prediction(my_NN, data_train_input, data_train_target, data_test_input, data_test_target,  size_training=train_size, num_epochs = Nmax, lr=my_lr, batch_size = my_batch_size)
    print(Data)
    plt.plot(range(Nmax), Data['error_round_train'], label='training')
    plt.plot(range(Nmax), Data['error_round_test'], label='testing')
    
    plt.xlabel('epoch')
    plt.ylabel(r'percent of false answer $\pm$ 0.4 ')
    plt.legend()
    
    plt.show()




def Opt_learning_rate(list_learning_rate):
    ''' 
    Evolution of the traing error according to various learning rate of the list list_learning_rate.
    '''
    for my_lr in list_learning_rate :
        Data = lib2.train(my_NN, data_train_input, data_train_target, size_training=train_size, num_epochs = Nmax, lr=my_lr, batch_size = my_batch_size)[1]
        plt.plot(range(Nmax), Data, label=str(my_lr))
    
    plt.xlabel('epoch')
    plt.ylabel(' round error')
    plt.legend(title='learning rate')
    plt.show()












