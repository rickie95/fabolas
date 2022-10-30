"""
    Performs experiments on SVMs trained on MNIST
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from datasets import load_mnist
from utils import save_results


def obj_function(configuration):

    size = None
    c = None
    gamma = None

    if len(configuration) == 3:
        c, gamma, size = configuration
        c = 10**c
        gamma = 10**gamma

    if len(configuration) == 2:
        c, gamma = 10**configuration

    assert (c is not None)
    assert (gamma is not None)

    dataset = load_mnist(training_size=size)

    c, gamma = 10**configuration
    grid = GridSearchCV(SVC(kernel="rbf"), {'C': [c], 'gamma': [gamma]}, n_jobs=-1, verbose=0, cv=5)
    grid.fit(dataset["X"], dataset["y"])
    return grid.best_score_


def load_prior(with_size=False):
    """
        Returns a dictionary holding rows referring to previuos runs

        "X": configurations, tuples of C and gamma values in log10 space (N, 2)
        "size": training set size (0, 1]
        "y": validation_score [0, 1]
        "c": log10 cost of the run, expressed in seconds
    """
    df = pd.read_csv("./results/fabolas/prior.csv")
    if not with_size:
        df = df.query("size > 0.20")

    return {
        "X": np.log10(np.array([df["C"], df["gamma"]]).T),
        "size": np.array(df["size"]).reshape(-1, 1),
        "y": np.array(df["validation_score"]).reshape(-1, 1),
        "c": np.array(df["time_s"]).reshape(-1, 1)
    }


def svm_mnist(method='random_search'):
    prior = None
    try:
        prior = load_prior()
    except:
        prior = generate_prior()

    assert(prior is not None)

    bounds = [(-10, 10), (-10, 10)]
    results = []

    if method == 'random_search':
        import random_search
        logging.info("Starting Random Search...")
        results, progress = random_search(obj_function, load_mnist)

    elif method == 'expected_improvement':
        from expected_improvement import ei
        logging.info("Starting Expected Improvement...")
        results, progress = ei(obj_function, prior, bounds)

    elif method == 'entropy_search':
        from entropy_search import es
        logging.info("Starting Entropy Search...")
        results, progress = es(obj_function, prior, bounds)

    elif method == 'fabolas':
        logging.info("Starting FABOLAS...")
        results, progress = fabolas(obj_function, data, prior, bounds)

    logging.info(f"Best value: {prior['y_best']}, with conf {prior['X_best']}")

    save_results(results, progress, method, "svm_mnist")


if __name__ == "__main__":
    logging.basicConfig(format='SVM_MNIST (%(process)s) - %(levelname)s - %(message)s', level=logging.INFO)
    svm_mnist(method='expected_improvement')
