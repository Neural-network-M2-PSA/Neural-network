from NetworkXor import Network
from Layer import Linear_layer, Activation_layer
from ActivationFunctions import *
from Loss2 import mse, mse_prime

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
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
out2 = net.predict(x_train)
for i in range(0,4):
    out2[i][0][0] = round(out[i][0][0])

print(out2)