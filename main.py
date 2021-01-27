import Librairie_v2 as lib2
import numpy as np
import pandas as pd

data = pd.read_csv('dataset.csv')
#print(data.head())
set_size = 1000
data_test_input = np.array(data[['cosTBz','R2','chiProb']][0:set_size])
data_test_target = np.array(data[['isSignal']][0:set_size])


my_layer1 = lib2.Linear(3,3)
my_layer2 = lib2.Sigmoid()
my_layer3 = lib2.Linear(3,1)
my_layer4 = lib2.Sigmoid()
my_NN = lib2.NeuralNet([my_layer1,my_layer2,my_layer3,my_layer4])

train(my_NN, data_test_input, data_test_target, batch_size = 32,size_training=set_size,num_epochs = 5000)

data_test_validation = np.array(data[['cosTBz','R2','chiProb']][set_size:set_size+100])
data_test_validation_target = np.array(data[['isSignal']][set_size:set_size+100])
print(prediction(my_NN,data_test_validation))