# FABOLAS: Fast Bayesian Optimization of Hyperparameters on Large Datasets

This repository replicates experiments from Klein, Falkner, Bartels, Henning, Hutter's "*Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets*" ([arXiv:1605.07079v2](https://arxiv.org/abs/1605.07079)).

## Experiments:
- [X] Showcase: grid search on SVM
- [X] SMV on 
  - [X] MNIST, 
  - [ ] Vehicle Registration,
  - [ ] Forest Cover Types 
- [X] CNN on:
  - [X] CIFAR-10,
  - [X] SVHN
- [ ] Deep Residual Network on CIFAR-10

### Showcase: Grid Search on SVM
File: ``svm-grid-search.py``

Just a playground, shows the impact of the dataset size on the hyperparameters search.
The script uses methods and API of ScikitLearn library, which provides a handy way to execute a grid search with cross validation.

Grid search is run on SVMs equipped with RBF kernel, searching for the best `C` and `gamma` couple that fit best the data.

### SVM on MNIST
File: ``svm-mnist.py``

Searches for the best couple of `C` and `gamma`, benchmarking three methods: Expected Improvement, Entropy Search and FABOLAS


### CNN on CIFAR10
File: ``cnn_cifar10.py``

Tries to find the best configuration choosing:
 - \# of filters for convolutional layer L1, L2, L3, mapped in log_2 space and bounded in [4, 9]
 - batch normalization
 - leanring rate, mapped in log_10, bounded in [-6, 0]
 
Methods tested: EI, ES, FABOLAS
 
 ### CNN on Street View House Numbers
File: ``cnn_svhn.py``

Tries to find the best configuration choosing:
 - \# of filters for convolutional layer L1, L2, L3, mapped in log_2 space and bounded in [4, 9]
 - batch normalization
 - leanring rate, mapped in log_10, bounded in [-6, 0]
 
 Methods tested: EI, ES, FABOLAS
