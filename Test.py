import Librairie as lb
import numpy as np

def test_MeanSquareError() :
    predicted = np.array([[1, 1, 1, 1, 1, 1],[2,4,5,6,8,1]])
    actual = np.array([[1, 1, 2, 1, 3, 1],[2,5,1,1,3,1]])
    
    mse = lb.MeanSquareError()
    
    print('test mse.loss : ', mse.loss(predicted, actual))
    print('test mse.grad : ', mse.grad(predicted, actual))
    
    

def test_LinearLayer_forward() :
    input = np.array([1, 1, 1, 1, 1, 1])
    lin = lb.Linear(6,4) #taille input et output
    print( lin.forward(input))
    
    
def test_LinearLayer_backward() :
    return 0

