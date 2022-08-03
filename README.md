# FABOLAS: Fast Bayesian Optimization of Hyperparameters on Large Datasets

This repository replicates experiments from Klein, Falkner, Bartels, Henning, Hutter's "*Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets*" ([arXiv:1605.07079v2](https://arxiv.org/abs/1605.07079)).

## Experiments:
- [X] Showcase: grid search on SVM
- [ ] SVM Grid on MNIST
- [ ] SMV with **no grid constraint** on MNIST, Vehicle Registration an Forest Cover Types 
- [ ] CNN on CIFAR-10 and SVHN
- [ ] Deep Residual Network on CIFAR-10

### Showcase: Grid Search on SVM
File: ``svm-grid-search.py``

Just a playground, shows the impact of the dataset size on the hyperparameters search.
The script uses methods and API of ScikitLearn library, which provides a handy way to execute a grid search with cross validation.

Grid search is run on SVMs equipped with RBF kernel, searching for the best `C` and `gamma` couple that fit best the data.