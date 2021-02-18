# Description of our report's figure

The figures of our report are reproductible withe the following parameters :

### Histogram of predictions
Article_and_figures/hist_ne1000_lr0.1_l631_s1.png
- train_size = 3000
- test_size = 1500
- my_batch_size = 100
- my_lr = 0.1
- my_num_epochs = 1000
- seed = 1
- Network = Linear(6,4)-Tanh()-Linear(4,2)-Tanh()-Linear(2,1)-Sigmoid()

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


