import numpy as np

from Network import Network
from Layer import Linear_layer, Activation_layer
from ActivationFunctions import tanh, tanh_prime
from Loss import Loss, MeanSquareError

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(Linear_layer(2, 3))
net.add(Activation_layer(tanh, tanh_prime))
net.add(Linear_layer(3, 1))
net.add(Activation_layer(tanh, tanh_prime))

# train
net.use(Loss, MeanSquareError)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)