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


import matplotlib.pyplot as matplot


training_sizes = [128, 16]

results = {}

def main():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    for training_size in training_sizes:
        indices = random.sample(range(1, train_x.shape[0]), floor(train_x.shape[0]/training_size))
        results[training_size] = np.zeros((20, 20))

        for c, gamma in np.ndindex((20,20)):
            classifier = svm.SVC(kernel='rbf', C=pow(e, c - 10), gamma=pow(e, gamma - 10))
            classifier.fit(train_x[indices], train_y[indices])
            predictions = classifier.predict(test_x)
            score = accuracy_score(predictions, test_y)
            results[training_size][c - 10][gamma - 10] = score
            print(f"|{c - 10}\t{gamma - 10}| Score: {score}")

        np.savetxt(f"{training_size}.csv", results[training_size])
    


if __name__ == "__main__":
    main()