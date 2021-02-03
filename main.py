from Network2 import Network
from layer2 import Linear_layer, Activation_layer
from ActivationFunctions import *
from Loss2 import mse, mse_prime


import pandas as pd

data = pd.read_csv('D:\data_train.csv')
data_test = pd.read_csv('D:\data_train.csv')
#print(data.head())
set_size = 258

# training data
x_train = np.array(data[['chiProb','Ks_distChi','m_KSpipi_norm','Mbc_norm']][0:set_size])
y_train = np.array(data[['isSignal']][0:set_size])


# network
net = Network()

net.add(Linear_layer(4, 4))
net.add(Activation_layer(tanh, tanh_prime))

net.add(Activation_layer(sigmoid, sigmoid_prime))
net.add(Linear_layer(4, 1))



# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=258, learning_rate=0.01)

# test
x_test = np.array(data_test[['chiProb','Ks_distChi','m_KSpipi_norm','Mbc_norm']][0:set_size])
y_test = np.array(data_test[['isSignal']][0:set_size])
out = net.predict(x_test)
#print(out)
#print("Type d'entrée", type(x_train))
#print("Type de sortie", type(out))
#print("Type y_train", type(y_train))

#print("shape d'entrée", x_train.shape)
print("shape de sortie", out.shape)
print("shape y_train", y_train.shape)


nb_true=0
nb_false=0

min = 0
tmp = 10
for i in range(258):

    #print("____________________________________________________")
    #print("step: ", i)
    #print("Prédiction: ", round(out[i][0]))
    #print("Valeur réelle: ", round(y_test[i][0]))
    if(round(out[i][0]) == round(y_test[i][0])):
        nb_true=nb_true+1
        print("TRUE")
    if (round(out[i][0]) != round(y_test[i][0])):
        print("FALSE")
        nb_false=nb_false+1
    print("____________________________________________________")


print("nb_true", nb_true)
print("nb_false", nb_false)
print("Pourcentage juste: ", nb_true/(nb_true+nb_false)*100, "%")

#print(y_train)



