from Network2 import Network
from Layer import Linear_layer, Activation_layer
from ActivationFunctions import *
from Loss2 import mse, mse_prime


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
#net.add(Activation_layer(relu, relu_prime))
net.add(Linear_layer(3, 1))
net.add(Activation_layer(tanh, tanh_prime))
#net.add(Activation_layer(relu, relu_prime))


# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=300, learning_rate=0.1)

# test
out = net.predict(x_train)
#print(out)
#print("Type d'entrée", type(x_train))
#print("Type de sortie", type(out))
#print("Type y_train", type(y_train))

#print("shape d'entrée", x_train.shape)
print("shape de sortie", out.shape)
print("shape y_train", y_train.shape)

for i in range(301):
    print("Prédiction: ", out[i][0])
    print("Valeur réelle: ", y_train[i][0])
    print("____________________________________________________")


#print(y_train)

out2 = net.predict(x_train)
for i in range(0,4):
    out2[i][0][0] = round(out[i][0][0])

print(out2)

