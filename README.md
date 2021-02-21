# M2-PSA project : Make your own neural network with Numpy
2020-2021

BROUILLARD Alizée, alizeebrouillard020198@gmail.com

BRUANT Quentin, quentinbruant92@gmail.com

GODINAUD Leila, leila.godinaud@gmail.com


## Abstract

The goal of this project is to build from scratch a neural network using only numpy.
After we test it, we develop few algorithm in order to optimize the hyper-parameters.
We work on the learning rate, batch size, number of neurons, number of hidden layers and the number of epoch.
All these tests reveal some imperfections (minimum local, computing time) that we try to improve.
For these we add two options : an adaptative learning rate (decay steps) and an alternative way to train :
the simultaneous learning.
The data set we use, allowed to answer to an praticle problem of particle physics :
how to separate Signal from Noise on an observed event ?
We also prepare these data set thanks to pre-processing algorithms.
After constraining the varibles in the [0,1] interval when needed, we construct the histogramms of these variables, then we build the cumulative function of the signal only.
By assigning the variables of a given event to its value in their respective cumulative functions, we separate the signal and the noise.

## Structure

The structure of our project is the folowing, please go inside each file to have its extensive description.

- file : Function_Tests
    - Test.py

- file : Main
    - main.py
    - SimultaneousNN.py

- file : Neural_Network_Library
    - activation_functions.py
    - error_round.py
    - layer.py
    - loss.py
    - neural_network.py
    - optimizer.py
    - user.py

- file : Optimization
    - Opt_network.py
    - Opt_num_epoch.py



## User's advices

### Standard use : write, train and predict with a Neural Network

Basic use of the neural network algorithm is presented in main.py with two example :
- A basic training with the default function loss and optimizer
- An alternative training with the adaptive learning rate (DecaySGD() as the optimizer)

The results are presented as an histogram of the prediction and a learning curve for each case.


An other possibility is to use SimultaneousNN.py in order to train an neural network with simultaneous learning.
You will find in this file the definition of this alternative training function and an example of its utilisation.

To compare the performances of our NN, we made a very basic NN with keras of tensorflow. This NN has the same 
properties as ours: same number of layers, activation functions, training size, testing size and learning rate. 
In the compile method, we choose the mean square error for the loss function, SGD for the optimizer with the
learning rate of 0.01. If we fit this model with a number of epochs egal to 5000 and a batch\_size of 100,
we obtain an accuracy of 0.86. This is coherent with ours previous results.

Link of the colaboratory of the keras NN : 
https://colab.research.google.com/drive/14BAD2qQf6H6PiBS18YoFzTYiymhBXqI4#scrollTo=bR3puNEJLJp8

### Optimization use

If you want to optimize a neural network, our optimisation functions are in two file.

- Opt_network.py
    - test_nbr_neuron(list_test) : tests different number of neurons
    - test_nbr_layer(list_test, n_neuron) : tests different number of hidden layers

- Opt_num_epoch.py
    - Opt_nbr_epoch() : produce a learning curve (training and testing error according to eppoch) in order to choose the optimal maximal number of epoch
    - Opt_learning_rate(list_learning_rate) : tests different learning rate


## References

- A. Boucaud, Cooking a simple neural network library, April, 2019, https://gitlab.in2p3.fr/ccin2p3-support/formations/workshops-gpu/04-2019/deep-learning/-/blob/master/notebooks/simple_nn_library.ipynb
 
- O.Aflak, Neural Network from scratch in Python, 15 November, 2018, https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

- A.Atakulreka and D.Sutivong, Avoiding Local Minima in Feedforward Neural Networks by Simultaneous Learning, 2007 , http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.100.2375&rep=rep1&type=pdf













