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









## User's advice

### Standard use : write, train and predict with a Neural Network

### Optimization use




