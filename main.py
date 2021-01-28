import numpy as np

from Network import Network
from Layer import Linear_layer, Activation_layer
from ActivationFunctions import tanh, tanh_prime
from Loss import Loss, MeanSquareError
from Loss2 import loss, grad, mse, mse_prime


import pandas as pd

data = pd.read_csv('D:\dataset.csv')
#print(data.head())
set_size = 100

# training data
x_train = np.array(data[['cosTBz','R2','chiProb']][0:set_size])
y_train = np.array(data[['isSignal']][0:set_size])

# network
net = Network()
net.add(Linear_layer(3, 3))
net.add(Activation_layer(tanh, tanh_prime))
net.add(Linear_layer(3, 1))
net.add(Activation_layer(tanh, tanh_prime))


# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=300, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)









