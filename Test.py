import Librairie_v2 as lib2
import numpy as np
import pandas as pd

def test_MeanSquareError() :
    predicted = np.array([[1, 1, 1, 1, 1, 1],[2,4,5,6,8,1]])
    actual = np.array([[1, 1, 2, 1, 3, 1],[2,5,1,1,3,1]])
    
    mse = lib2.MeanSquareError()
    
    print('test mse.loss : ', mse.loss(predicted, actual))
    print('test mse.grad : ', mse.grad(predicted, actual))
    ''' OK '''
    
    

def test_LinearLayer() :
    input = np.array([[1, 1, 1, 1, 1, 1]])
    grad = np.array([[0.5,0.2,0.3,0.1]])
    lin = lib2.Linear(6,4) #taille input et output
    print('Y', lin.forward(input))
    print( 'grad',lin.backward(grad))
    print('grad_w',lin.grad_w)
    print('grad_b',lin.grad_b)
    ''' OK '''
    
def test_NeuralNet() :
    my_layer1 = lib2.Linear(3,2)
    my_layer2 = lib2.Tanh()
    my_NN = lib2.NeuralNet([my_layer1,my_layer2])
    
    input = np.array([[1,2,3],[4,5,6]])
    grad =  np.array([[0.5,0.2],[0.1,0.3]])
    
    
    print('forward', my_NN.forward(input))
    print('backward', my_NN.backward(grad))
    '''OK'''
    
    

def test_train_prediction() :
    my_layer1 = lib2.Linear(3,2)
    my_layer2 = lib2.Tanh()
    my_NN = lib2.NeuralNet([my_layer1,my_layer2])
    
    input = np.array([[1,2,3],[4,5,6]])
    target = np.array([[0.5,0.2],[0.1,0.3]])
    
    lib2.train(my_NN, input, target, batch_size = 1,size_training=2)
    #attention il faut que size_training <= à nbre de ligne dans data
    
    input_predict = np.array([[1,1,4],[0.5,2,4]])
    print(lib2.prediction(my_NN,input_predict))
    ''' OK '''


def test_XOR() :
    my_layer1 = lib2.Linear(2,3)
    my_layer2 = lib2.Tanh()
    my_layer3 = lib2.Linear(3,1)
    my_layer4 = lib2.Sigmoid()
    #my_layer3 = lib2.Arondi()
    my_NN = lib2.NeuralNet([my_layer1,my_layer2,my_layer3,my_layer4])
    
    input =np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    target = np.array([[0], [1], [1], [0]])
    
    lib2.train(my_NN, input, target, batch_size = 1,size_training=4,num_epochs= 10000)
    #attention il faut que size_training <= à nbre de ligne dans data
    
    input_predict = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    print(lib2.prediction(my_NN,input_predict))
    ''' OK '''
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    