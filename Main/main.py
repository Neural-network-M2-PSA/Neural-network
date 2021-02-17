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

## Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## importations
import Neural_Network_Library.layer as Layer
import Neural_Network_Library.activation_functions as ActivationFunctions
import Neural_Network_Library.neural_network as Neural_network
import Neural_Network_Library.user as User
import Neural_Network_Library.optimizer as Optimizer
import Neural_Network_Library.error_round as error_round2


import Optimization.Opt_network
import Optimization.Opt_num_epoch
import Function_Tests.Test

plt.close()

## Parameters' choice

'''Seed'''
np.random.seed(1)

'''size of the training set and the testing set '''
train_size = 3000
test_size = 1500

'''Maximal number of epochs '''
my_num_epochs = 500

'''size of the batch'''
my_batch_size=100

''' learning rate'''
my_lr=0.001

my_initial_lr = 0.1
my_decay_coeff =1/200

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



## Importation of the training and testing data
data_training_path='Neural-network2/Data/data_train+.csv'
data_test_path='Neural-network2/Data/data_test+.csv'

Data_train = pd.read_csv(data_training_path)
Data_test = pd.read_csv(data_test_path)


param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']

'''training set'''
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

'''testing set'''
data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])



## Basic use : training and testing

print('basic example of utilisation :')

'''training'''
chi2_list, error_list = User.train(my_NN, data_train_input, data_train_target , num_epochs = my_num_epochs, optimizer = Optimizer.SGD(lr=my_lr), batch_size=my_batch_size)

plt.plot(range(my_num_epochs), chi2_list)
plt.xlabel('epoch')
plt.ylabel('mean squared error')
plt.title('Evolution of the training error : basic example')

'''testing '''
data_test_prediction = User.prediction(my_NN,data_test_input)

error = error_round2.error_round(data_test_prediction, data_test_target)
print('% false error for testing set : ', error)


'''histogram of predictions'''
plt.figure()
S = data_test_prediction[data_test_target==1]
B = data_test_prediction[data_test_target==0]

kwargs = dict(histtype="stepfilled", alpha=0.5, bins=25)

plt.hist(S, **kwargs, label ='Signal')
plt.hist(B, **kwargs, label='Noise')
plt.legend(loc="upper center")
plt.xlabel('predictions for testing set')
plt.ylabel('number of answer')
plt.title('Basic use')


## Alternative use : adaptative learning rate

print('\nAlternative example of utilisation (adaptative learning rate) :')

''' training'''

chi2_list, error_list = lib2.train(my_NN, data_train_input, data_train_target, num_epochs = my_num_epochs, batch_size=my_batch_size, optimizer = Optimizer.DecaySGD(initial_lr= my_initial_lr, decay_coeff= my_decay_coeff) )

plt.figure()
plt.plot(range(my_num_epochs), chi2_list)
plt.xlabel('epoch')
plt.ylabel('mean squared error')
plt.title('Evolution of the training error : adaptative learning rate')

plt.figure()
x=np.array(range(my_num_epochs))
plt.plot(x, my_initial_lr * np.exp(-my_decay_coeff *x))
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.title('Adaptative learning rate')


'''testing '''
data_test_prediction = User.prediction(my_NN,data_test_input)

error = error_round2.error_round(data_test_prediction, data_test_target)
print('% false error for testing set : ', error)


'''histogram of predictions and ROC curve '''
plt.figure()
S = data_test_prediction[data_test_target==1]
B = data_test_prediction[data_test_target==0]



plt.hist(S, **kwargs, label ='Signal')
plt.hist(B, **kwargs, label='Noise')
plt.legend(loc="upper center")
plt.xlabel('predictions for testing set')
plt.ylabel('number of answer')
plt.title('Alternative use')

print('\n')
plt.show()

