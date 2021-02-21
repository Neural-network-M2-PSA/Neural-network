# Description of our report's figure

The figures of our report are reproductible withe the following parameters :

### Learning curve
Article_and_figures/ne5e5_lr1e-4__l631_s1.png
- train_size = 3000
- test_size = 1500
- my_batch_size = 100
- my_lr = 1e-4
- my_num_epochs = 5e+5
- seed = 1
- Network = Linear(6,3)-Tanh()-Linear(3,1)-Sigmoid()


### Simultaneous learning
Article_and_figures/SimNN_l6421_lr1e-4_bs100_s1.png
- train_size = 3000
- test_size = 1500
- my_batch_size = 100
- my_lr = 1e-4
- my_num_epochs = 1e+4
- seed = 1
- Network = Linear(6,4)-Tanh()-Linear(4,2)-Tanh()-Linear(2,1)-Sigmoid()
- Simultaneous learning


### Histogram of optimised predictions
Article_and_figures/hist_optimal2.png
- train_size = 3000
- test_size = 1500
- my_batch_size = 100
- my_initial_lr = 0.1
- my_decay_coeff = 1/50000
- my_num_epochs = 250000
- seed = 1
- Network = Linear(6,4)-Tanh()-Linear(4,4)-Tanh()-Linear(4,4)-Tanh()-Linear(4,1)-Sigmoid()
- Simultaneous learning
- Decay SGD


