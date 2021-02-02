import Librairie_v3 as lib3
import numpy as np
import pandas as pd

def test_MeanSquareError() :
    predicted = np.array([[1, 1, 1, 1, 1, 1],[2,4,5,6,8,1]])
    actual = np.array([[1, 1, 2, 1, 3, 1],[2,5,1,1,3,1]])
    
    mse = lib3.MeanSquareError()
    
    print('test mse.loss : ', mse.loss(predicted, actual))
    print('test mse.grad : ', mse.grad(predicted, actual))
    ''' OK '''
    
    

def test_LinearLayer() :
    input = np.array([[1, 2, 1, 8, 5, 1]])
    grad = np.array([[0.5,0.2,0.3,0.1]])
    lin = lib3.Linear(6,4) #taille input et output
    print('Y', lin.forward(input))
    print( 'grad',lin.backward(grad))
    print('grad_w',lin.grad_w)
    print('grad_b',lin.grad_b)
    ''' OK '''
    
def test_NeuralNet() :
    my_layer1 = lib3.Linear(3,2)
    my_layer2 = lib3.Tanh()
    my_NN = lib3.NeuralNet([my_layer1,my_layer2])
    
    input = np.array([[1,2,3],[4,5,6]])
    grad =  np.array([[0.5,0.2],[0.1,0.3]])
    
    
    print('forward', my_NN.forward(input))
    print('backward', my_NN.backward(grad))
    '''OK'''
    
    

def test_train_prediction() :
    my_layer1 = lib3.Linear(3,2)
    my_layer2 = lib3.Tanh()
    my_NN = lib3.NeuralNet([my_layer1,my_layer2])
    
    input = np.array([[1,2,3],[4,5,6]])
    target = np.array([[0.5,0.2],[0.1,0.3]])
    
    lib3.train(my_NN, input, target,size_training=2)
    #attention il faut que size_training <= à nbre de ligne dans data
    
    input_predict = np.array([[1,1,4],[0.5,2,4]])
    print(lib3.prediction(my_NN,input_predict))
    ''' OK '''


def test_XOR() :
    my_layer1 = lib3.Linear(2,3)
    my_layer2 = lib3.Tanh()
    my_layer3 = lib3.Linear(3,1)
    my_layer4 = lib3.Tanh()
    #my_layer4 = lib3.Arondi()
    my_NN = lib3.NeuralNet([my_layer1,my_layer2,my_layer3,my_layer4])
    
    input =np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    target = np.array([[0], [1], [1], [0]])
    
    lib3.train(my_NN, input, target,size_training=4, lr=0.1)
    #attention il faut que size_training <= à nbre de ligne dans data
    
    input_predict = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    print(lib3.prediction(my_NN,input_predict))
    ''' OK '''
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    