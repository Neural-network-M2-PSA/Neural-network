"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""
"""_______SimultaneousNN_____"""

'''
Improved method thanks to simultaneous learning
See : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.100.2375&rep=rep1&type=pdf
for more informations
'''


## importations

import numpy as np
from numpy import ndarray as Tensor
import pandas as pd

import os as os
path_ini = os.getcwd()
path= path_ini[:-4]+'Neural_Network_Library'
os.chdir(path)

import optimizer as OptimizerClass
import user as User
import layer as Layer
import activation_functions as ActivationFunctions
import neural_network as Neural_network
import error_round as Error_round
import loss as Loss

import matplotlib.pyplot as plt
plt.close()

## Parameters' choice
'''seed '''
np.random.seed(1)

'''Maximal number of epochs '''
Nmax = 500

'''size of the training set and the testing set '''
train_size = 3000
test_size = 1500

'''size of the batch'''
my_batch_size=100

''' learning rate'''
my_lr=0.0001

'''importation of the training and testing data'''
os.chdir(path_ini[:-4])
Data_train = pd.read_csv('Data/data_train.csv')
Data_test = pd.read_csv('Data/data_test.csv')

param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])


## Modified function

def train_simultaneousNN( inputs_train: Tensor, targets_train: Tensor, loss: Loss.Loss = Loss.MeanSquareError(), optimizer: OptimizerClass.Optimizer = OptimizerClass.SGD(), num_epochs: int = 5000, batch_size : int = 32) -> tuple:
    
    size_training = inputs_train.shape[0]
    Result_chi2 = [[],[],[],[],[],[],[],[],[]]
    list_epoch = np.array(range(10,50,5))/100 * num_epochs
    
    '''initialisation des 9 NN''' #verifier question seed()
    list_net =[]
    for i in range(9) :
        layers=[]
        layers.append(Layer.Linear(6,4))
        layers.append(ActivationFunctions.Tanh())
        layers.append(Layer.Linear(4,2))
        layers.append(ActivationFunctions.Tanh())
        layers.append(Layer.Linear(2,1))
        layers.append(ActivationFunctions.Sigmoid())
        list_net.append(Neural_network.NeuralNet(layers))
    
    destroyed_NN=[]
    nbr_batch=size_training//batch_size
    
    ''' training des 9 NN'''
    for epoch in range(num_epochs):
        
        for k in range(9) :
            if k not in destroyed_NN :
                Chi2_train =0
                
                for i in range(0,size_training,batch_size):
                    
                    # 1) feed forward
                    y_actual = list_net[k].forward(inputs_train[i:i+batch_size])
                    
                    # 2) compute the loss and the gradients
                    Chi2_train += loss.loss(targets_train[i:i+batch_size],y_actual)
                    grad_ini = loss.grad(targets_train[i:i+batch_size],y_actual)
                    
                    # 3)feed backwards
                    grad_fini = list_net[k].backward(grad_ini)
                    
                    # 4) update the net 
                    optimizer.step(list_net[k], n_epoch = epoch)
                    
                Chi2_train= Chi2_train/nbr_batch
                Result_chi2[k].append(Chi2_train)
        
        '''Supression du NN le moins efficace '''
        if epoch in list_epoch :
            Comparaison=[[],[]]
            for k in range(9):
                if k not in destroyed_NN :
                    ErrorSlope = np.polyfit(np.array(range(epoch-49,epoch)), Result_chi2[k][-50:-1],1)[0]
                    MixedError = Result_chi2[k][-1] * (1-np.arctan(ErrorSlope)/(np.pi/2))
                    Comparaison[0].append(k)
                    Comparaison[1].append(MixedError)
            
            k = Comparaison[0][Comparaison[1].index(max(Comparaison[1]))]
            destroyed_NN.append(k)
        
        if epoch % 100 == 0:
            print('epoch : '+str(epoch)+"/"+str(num_epochs)+"\r", end="")
    
    for k in range(9):
        if k not in destroyed_NN :
            my_NN = list_net[k]
    return my_NN, Result_chi2
    



## User's function

Results = train_simultaneousNN(data_train_input, data_train_target, num_epochs = Nmax, optimizer = OptimizerClass.SGD(lr = my_lr), batch_size=my_batch_size)


for k in range(9) :
    Y = Results[1][k]
    X = range(len(Y))
    plt.plot(X,Y)
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.title('Simultaneous NN : training curve')



'''testing '''
data_test_prediction = User.prediction(Results[0],data_test_input)

error = Error_round.error_round(data_test_prediction, data_test_target)
print('% of false error for testing set : ', error)


'''histogram of predictions'''
plt.figure()
S = data_test_prediction[data_test_target==1]
B = data_test_prediction[data_test_target==0]

kwargs = dict(histtype="stepfilled", alpha=0.5, bins=25)

plt.hist(S, **kwargs, label ='Signal')
plt.hist(B, **kwargs, label='Noise')
plt.legend(loc="upper center")
plt.xlabel('Predictions for testing set')
plt.ylabel('Number of answer')
plt.title('Simultaneous NN : histogram of predictions')

plt.show()

