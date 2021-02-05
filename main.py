import Librairie_v2 as lib2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.close()

import random as rd
rd.seed(1)

Data_train = pd.read_csv('data_train.csv')
#print(Data_train.head())
Data_test = pd.read_csv('data_test.csv')
'''
description data :
data_train.csv -> 3000
data_test.csv -> 1500
data_validation.csv -> 1500
param = 'cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm', 'isSignal'

'''
train_size = 3000
param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])


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

lib2.train_graph(my_NN, data_train_input, data_train_target , size_training=train_size, num_epochs = 5000, lr=0.1, batch_size=100)

test_size = 1500
data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])
data_test_prediction = lib2.prediction(my_NN,data_test_input)

results = pd.DataFrame()
results['NN']=data_test_prediction.T[0]
results['validation']=data_test_target
print(results)
print('loss data_test',np.mean((data_test_prediction-data_test_target)**2))
A = np.round(data_test_prediction)
print('% erreur avec arrondi prediction', np.sum( A[A != data_test_target])*100/test_size)
















