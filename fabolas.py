""" FABOLAS: Fast Bayesian Optimization for Large Datasets 

    1. Initialize dataset D_0
    2. for t = 1, 2, ...
    3.     fit GP models f(x, s) and c(x, s) on dataset D_{t-1}
    4.     evaluate y_t measuring its cost z_t
    5.     D_t = D_{t-1} + {(x_t, s_t, y_t, z_t)}
    6.     choose the estimate of x based on the predicted loss for 
            the entire dataset (s=1) of all x_1, x_2, ..., x_t
    7. end for

"""

from cmath import e
from math import floor
import random
from statistics import median
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn import svm
import seaborn as sb
import numpy as np
import time


import matplotlib.pyplot as plt

def obj_function(configuration, data):
    """
    - configuration: dictionary of parameters
        eg.
        {
            "C" : 10,
            "gamma" : 0.001
        }
    - data: 
    """
    startTime = time.time()

    # Do thing
    regressor = svm.SVC(C=configuration["C"], gamma=configuration["gamma"], kernel="rbf").fit(data["X"], data["y"])
    score = regressor.score(data["X_validation"], data["y_validation"])

    executionTime = time.time() - startTime

    return score, executionTime


def init_dataset(obj_function, parameters, dataset, sizes, initial_points=10):
    """
    Evaluates the obj function with K samples of the parameter space for each subset size.

    - obj_function: func(configuration, size)
        Function to optimize
    - parameters: dict of parameters
        An entry for each parameter, including its bounds
        eg.
        {   
            "A" : { "max" : 10, "min" : -10 },
            "B" : { "max" : 1, "min" : 0 }
        }
    - dataset:
    - sizes: fractions of original dataset to consider
        eg. [128, 64, 2, 1] -> 1/128, 1/64, 1/2, 1
    """

    # Generate initial configurations
    configurations = []
    for i in range(initial_points):
        config = {}
        for p in parameters:
            config[p] = random.choice(range(parameters[p]["min"], parameters[p]["max"] + 1))
        configurations.append(config)

    # For each training subset size, evaluate the function over all generated configurations
    for s in sizes:
        for c in configurations:
            score, time = obj_function(configuration, data)
            dataset.add(configuration, training_size, score, time)

    return dataset

def acquisition_function():
    # TODO
    training_size = 128
    configuration = {"C" : 10, "gamma": 0.00001}
    return configuration, training_size

def load_dataset():
    
    return data

def main():
    #dataset = init_dataset(obj_function, parameters, dataset, dataset_subsets_size)
    dataset = []
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    configuration, training_size = acquisition_function()

    indices = random.sample(range(1, train_x.shape[0] - 10000), floor((train_x.shape[0] - 10000)/training_size))

    data = {}
    # 10k examples are reserved for validation
    data["X_validation"], data["y_validation"] = train_x[:-10000], train_y[:-10000]
    data["X"], data["y"] = train_x[indices], train_y[indices]


    parameters = {   
            "C" : { "max" : 10, "min" : -10 },
            "gamma" : { "max" : 10, "min" : -10 }
        }
    
    dataset = init_dataset(obj_function, parameters, data, [64, 32, 16, 8])



    # fit Gaussian Process Regressors for f(x, s) and c(x, s) based on data
    # f(x, s) is the score of the regressor/classifier
    # c(x, s) is the cost (meaning time) of the function evaluation given configuration and data
    


if __name__ == "__main__":
    main()