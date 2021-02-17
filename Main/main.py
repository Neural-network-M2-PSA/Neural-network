"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______main_____"""

'''
Basic use of the neural network algorithm.
'''

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importation
import Neural_Network_Library.layer as Layer
import Neural_Network_Library.activation_functions as ActivationFunctions
import Neural_Network_Library.neural_network as Neural_network
import Neural_Network_Library.user as User
import Neural_Network_Library.Data

import Optimization.Opt_network
import Optimization.Opt_num_epoch
import Function_Tests.Test

plt.close()

'''
Seed
'''
np.random.seed(1)

# Parameters' choice

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
my_layer1 = Layer.Linear(6,5)
my_layer2 = ActivationFunctions.Tanh()
my_layer3 = Layer.Linear(5,4)
my_layer4 = ActivationFunctions.Tanh()
my_layer5 = Layer.Linear(4,3)
my_layer6 = ActivationFunctions.Tanh()
my_layer7 = Layer.Linear(3,2)
my_layer8 = ActivationFunctions.Tanh()
my_layer9 = Layer.Linear(2,1)
my_layer10 = ActivationFunctions.Sigmoid()
my_NN = Neural_network.NeuralNet([my_layer1, my_layer2, my_layer3, my_layer4, my_layer5, my_layer6, my_layer7, my_layer8, my_layer9, my_layer10])



# Importation of the training and testing data

Data_train = pd.read_csv('data_train.csv')
Data_test = pd.read_csv('data_test.csv')


param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']

'''training set'''
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

'''testing set'''
data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])



# Training

'''training'''
chi2_list, error_list = User.train(my_NN, data_train_input, data_train_target , size_training=train_size, num_epochs = my_num_epochs, lr=my_lr, batch_size=my_batch_size)



plt.plot(range(my_num_epochs), error_list)
plt.xlabel('epoch')
plt.ylabel('round error')
plt.title('Evolution of the training error')
plt.show()

# Testing

data_test_prediction = User.prediction(my_NN,data_test_input)

'''comparison between the predictions and the true results '''
results = pd.DataFrame()
results['NN']=data_test_prediction.T[0]
results['validation']=data_test_target
print(results)



print('% erreur avec arrondi prediction',error_list[-1])
















