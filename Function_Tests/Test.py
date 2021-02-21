"""
M2-PSA project
2020-2021
BROUILLARD Alizée, alizeebrouillard020198@gmail.com
BRUANT Quentin, quentinbruant92@gmail.com
GODINAUD Leila, leila.godinaud@gmail.com

"""

"""_______Test_____"""
'''
This file contains preliminary tests to verify the function MeanSquareError, the class linear layer, the class neural network, 
the functions train and prediction. We have also a test for a XOR to make sure that our neural network
learns something. 
'''


# Library
import numpy as np

import os as os
path_ini = os.getcwd()

path= path_ini[:-14]+'Neural_Network_Library' #changement dans cette section
os.chdir(path)

# Imports
import loss as Loss
import layer as Layer
import activation_functions as ActivationFunctions
import neural_network as Neural_network
import user as User

'''
## Imports

import Neural_Network_Library.loss as Loss
import Neural_Network_Library.layer as Layer
import Neural_Network_Library.activation_functions as ActivationFunctions
import Neural_Network_Library.neural_network as Neural_network
import Neural_Network_Library.user as User

'''


def test_MeanSquareError() :
    predicted = np.array([[1, 1, 1, 1, 1, 1],[2,4,5,6,8,1]])
    actual = np.array([[1, 1, 2, 1, 3, 1],[2,5,1,1,3,1]])
    
    mse = Loss.MeanSquareError()
    
    print('test mse.loss : ', mse.loss(predicted, actual))
    print('test mse.grad : ', mse.grad(predicted, actual))
    ''' OK '''
    

def test_LinearLayer() :
    input = np.array([[1, 1, 1, 1, 1, 1]])
    grad = np.array([[0.5,0.2,0.3,0.1]])
    lin = Layer.Linear(6,4) #taille input et output
    print('Y', lin.forward(input))
    print( 'grad',lin.backward(grad))
    print('grad_w',lin.grad_w)
    print('grad_b',lin.grad_b)
    ''' OK '''
    
def test_NeuralNet() :
    my_layer1 = Layer.Linear(3,2)
    my_layer2 = ActivationFunctions.Tanh()
    my_NN = Neural_network.NeuralNet([my_layer1,my_layer2])
    
    input = np.array([[1,2,3],[4,5,6]])
    grad =  np.array([[0.5,0.2],[0.1,0.3]])
    
    
    print('forward', my_NN.forward(input))
    print('backward', my_NN.backward(grad))
    '''OK'''
    
    

def test_train_prediction() :
    my_layer1 = Layer.Linear(3,2)
    my_layer2 = ActivationFunctions.Tanh()
    my_NN = Neural_network.NeuralNet([my_layer1,my_layer2])
    
    input = np.array([[1,2,3],[4,5,6]])
    target = np.array([[0.5,0.2],[0.1,0.3]])
    
    User.train(my_NN, input, target, batch_size = 1)
    #By careful, we must have size_training = number of rows in our data
    
    input_predict = np.array([[1,1,4],[0.5,2,4]])
    print(User.prediction(my_NN,input_predict))
    ''' OK '''


def test_XOR() :
    my_layer1 = Layer.Linear(2,3)
    my_layer2 = ActivationFunctions.Tanh()
    my_layer3 = Layer.Linear(3,1)
    my_layer4 = ActivationFunctions.Sigmoid()
    #my_layer3 = lib2.Arondi()
    my_NN = Neural_network.NeuralNet([my_layer1,my_layer2,my_layer3,my_layer4])
    
    input =np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    target = np.array([[0], [1], [1], [0]])
    
    User.train(my_NN, input, target, batch_size = 1,,num_epochs= 1000)
    # By careful, we must have size_training = number of rows in our data
    
    input_predict = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    print(User.prediction(my_NN,input_predict))
    ''' OK '''
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
