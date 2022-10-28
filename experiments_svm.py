"""
    Performs experiments on SVMs trained on MNIST
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from datasets import load_mnist

import pandas as pd
import numpy as np

import logging


def obj_function(configuration, dataset=None):
    if dataset is None:
        dataset = load_mnist(training_size=1/64)

    c, gamma = 10**configuration
    grid = GridSearchCV(SVC(kernel="rbf"), {'C': [c], 'gamma': [gamma]}, n_jobs=-1, verbose=1, cv=5)
    grid.fit(dataset["X"], dataset["y"])
    return 1 - grid.best_score_


def load_prior():
    """
        Returns a dictionary holding rows referring to previuos runs

        "X": configurations, tuples of C and gamma values in log10 space (N, 2)
        "size": training set size (0, 1]
        "y": validation_error [0, 1]
        "c": log10 cost of the run, expressed in seconds
    """
    df = pd.read_csv("./results/fabolas/prior.csv")
    df = df.query("size > 0.20")

    return {
        "X": np.log10(np.array([df["C"], df["gamma"]]).T),
        "size": np.array(df["size"]).reshape(-1, 1),
        "y": 1 - np.array(df["validation_score"]).reshape(-1, 1),
        "c": np.array(df["time_s"]).reshape(-1, 1)
    }


if __name__ == "__main__":

    logging.basicConfig(format='SVM_MNIST (%(process)s) - %(levelname)s - %(message)s', level=logging.INFO)
    # TODO: some steps to perform:
    # load prior
    # set bounds
    # perform hyper opt
    # print/save results

    prior = load_prior()
    # prior = generate_prior()

    bounds = [(-10, 10), (-10, 10)]
    results = []

    method = 'expected_improvement'

    if method == 'random_search':
        import random_search
        random_search(obj_function, load_mnist)
    elif method == 'expected_improvement':
        from expected_improvement import ei
        results = ei(obj_function, prior, bounds)
    elif method == 'entropy_search':
        import entropy_search
        entropy_search(obj_function, data, prior, bounds)
        pass
    elif method == 'fabolas':
        fabolas(obj_function, data, prior, bounds)

    print(f"Best value: {prior['y_best']}, with conf {prior['X_best']}")