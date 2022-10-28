import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from acquisitions import expected_improvement

import logging


def ei(obj_function, prior, bounds=None) -> np.array:
    """
        Perform bayesian optimization of `obj_function` over `data`, using 
        `prior` as baseline.

        ### Parameters:
        - obj_function a method which accepts as parameters:
            - X_candidate: np.array(1, D) candidate point/configuration to be evaluated
            - data: dataset
        - prior: dict of known points
            - "X" cordinates 
            - "y" function values
        - bounds: list of tuples holding optional bound used for candidates clip

        Returns the prior extended with the points evaluated during the optimization.
    """
    n_iter = 10

    for i in range(n_iter):
        print(f"---- EI: Iteration #{i+1} ----")

        X_candidate = get_candidate(prior, bounds)

        logging.info(f"Selected this candidate: {X_candidate}")

        if bounds is not None:
            for i, b in enumerate(bounds):
                X_candidate[i] = np.clip(X_candidate[i], b[0], b[1])

        logging.info("Evaluating function...")
        y_candidate = obj_function(X_candidate)

        logging.info(f"Function value: {y_candidate}")

        prior["X"] = np.vstack([prior["X"], X_candidate])
        prior["y"] = np.append(prior["y"], np.array([y_candidate]))

    prior["y_best"] = max(prior["y"])
    imax = np.argmax(prior["y"])
    prior["X_best"] = prior["X"][imax]

    return prior


def get_candidate(prior, bounds):
    n_gen_samples = 100

    cov = 1
    kernel = cov * Matern(nu=5/2, length_scale_bounds=(1e-10, 1e10))
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(prior["X"], prior["y"])

    X_samples = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_gen_samples, prior["X"].shape[1])
        )

    y_pred = gpr.predict(prior["X"])
    mean, covariance = gpr.predict(X_samples, return_cov=True)

    ei = expected_improvement(
        mean,
        covariance,
        y_pred
    )

    index = np.argmax(ei)
    return X_samples[index]
