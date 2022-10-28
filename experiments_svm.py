"""
    Performs experiments on SVMs trained on MNIST
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from datasets import load_mnist

import logging

def obj_function(configuration, dataset=None):
    if dataset is None:
        dataset = load_mnist()

    c, gamma = configuration
    grid = GridSearchCV(SVC(kernel="rbf"), {'C': c, 'gamma': gamma}, n_jobs=-1, verbose=3, cv=5)
    grid.fit(dataset["X"], dataset["y"])
    return grid.best_score_


if __name__ == "__main__":

    logging.basicConfig(format='SVM_MNIST (%(process)s) - %(levelname)s - %(message)s', level=logging.INFO)
    #TODO: some steps to perform:
    # load prior
    # set bounds
    # perform hyper opt
    # print/save results

    # prior = load_prior()
    prior = generate_prior()

    bounds = [(-10, 10),(-10, 10)]
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