"""_______Simultaneous Neural Network_____"""

'''

'''


## importations

import Neural_Network_Library.Librairie_v2 as lib2
#import Librairie_v2 as lib2
import numpy as np
from numpy import ndarray as Tensor
import pandas as pd

import matplotlib.pyplot as plt
plt.close()

## Parameters' choice


'''Maximal number of epochs '''
Nmax = 10000

'''size of the training set and the testing set '''
train_size = 3000
test_size = 1500

'''size of the batch'''
my_batch_size=100

''' learning rate'''
my_lr=0.0001

'''importation of the training and testing data'''
Data_train = pd.read_csv('data_train.csv')
Data_test = pd.read_csv('data_test.csv')

param = ['cosTBz', 'R2', 'chiProb', 'Ks_distChi', 'm_KSpipi_norm', 'Mbc_norm']
data_train_input = np.array(Data_train[param][:train_size])
data_train_target = np.array(Data_train[['isSignal']][:train_size])

data_test_input = np.array(Data_test[param][:test_size])
data_test_target = np.array(Data_test[['isSignal']][:test_size])


## Modified function

def train_simultaneousNN( inputs_train: Tensor, targets_train: Tensor, loss: lib2.Loss = lib2.MeanSquareError(), optimizer: lib2.Optimizer = lib2.SGD(), num_epochs: int = 5000, size_training : int =100, lr : float = 0.01, batch_size : int = 32) -> None:
    
    Result_chi2 = [[],[],[],[],[],[],[],[],[]]
    list_epoch = np.array(range(10,50,5))/100 * num_epochs
    
    '''initialisation des 9 NN''' #verifier question seed()
    list_net =[]
    for i in range(9) :
        layers=[]
        layers.append(lib2.Linear(6,4))
        layers.append(lib2.Tanh())
        layers.append(lib2.Linear(4,2))
        layers.append(lib2.Tanh())
        layers.append(lib2.Linear(2,1))
        layers.append(lib2.Sigmoid())
        list_net.append(lib2.NeuralNet(layers))
    
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
                    optimizer.lr = lr
                    optimizer.step(list_net[k])
                    
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
            print(epoch, destroyed_NN)
        
        if epoch % 100 == 0:
            print('epoch : '+str(epoch)+"/"+str(num_epochs)+"\r", end="")
    
    for k in range(9):
        if k not in destroyed_NN :
            my_NN = list_net[k]
    return my_NN, Result_chi2
    



## User's function

Results = train_simultaneousNN(data_train_input, data_train_target, num_epochs = Nmax, size_training=train_size, lr=my_lr, batch_size=my_batch_size)


for k in range(9) :
    Y = Results[1][k]
    X = range(len(Y))
    plt.plot(X,Y)
plt.ylabel('MSE error')
plt.xlabel('epoch')
plt.show()


data_test_prediction = lib2.prediction(Results[0], data_test_input)

results = pd.DataFrame()
results['NN']=data_test_prediction.T[0]
results['validation']=data_test_target
print(results)


print('round error traing', lib2.error_round(data_test_prediction,data_test_target))
print('MSE error training', np.mean((data_test_prediction-data_test_target)**2))

data_test_predictionbis = lib2.prediction(Results[0], data_train_input)
print('round error testing', lib2.error_round(data_test_predictionbis,data_train_target))


















