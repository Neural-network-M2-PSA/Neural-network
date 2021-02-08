"""_______Example of utilisation_____"""

'''
Basic use of the neural network algorithm
'''

## importation
import Librairie_v2 as lib2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.close()

import random as rd
rd.seed(1)

##Parameters' choice

'''size of the training set and the testing set '''
train_size = 3000
test_size = 1500

'''Maximal number of epochs '''
my_num_epochs = 500

'''size of the batch'''
my_batch_size=100

''' learning rate'''
my_lr=0.001

'''Construction of the neural network '''
my_layer1 = lib2.Linear(6,5)
my_layer2 = lib2.Tanh()
my_layer3 = lib2.Linear(5,4)
my_layer4 = lib2.Tanh()
my_layer5 = lib2.Linear(4,3)
my_layer6 = lib2.Tanh()
my_layer7 = lib2.Linear(3,2)
my_layer8 = lib2.Tanh()
my_layer9 = lib2.Linear(2,1)
my_layer10 = lib2.Sigmoid()
my_NN = lib2.NeuralNet([my_layer1, my_layer2, my_layer3, my_layer4, my_layer5, my_layer6, my_layer7, my_layer8, my_layer9, my_layer10])



## importation of the training and testing data

Data_train = pd.read_csv('data_train.csv')
Data_test = pd.read_csv('data_test.csv')

param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']

'''training set'''
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

'''testing set'''
data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])



## training

'''training'''
chi2_list, error_list = lib2.train(my_NN, data_train_input, data_train_target , size_training=train_size, num_epochs = my_num_epochs, lr=my_lr, batch_size=my_batch_size)



plt.plot(range(my_num_epochs), error_list)
plt.xlabel('epoch')
plt.ylabel('round error')
plt.title('Evolution of the training error')
plt.show()

## testing

data_test_prediction = lib2.prediction(my_NN,data_test_input)

'''comparison between the predictions and the true results '''
results = pd.DataFrame()
results['NN']=data_test_prediction.T[0]
results['validation']=data_test_target
print(results)



print('% erreur avec arrondi prediction',error_list[-1])















