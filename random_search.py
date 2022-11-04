import logging
import time

import numpy as np


def rs(obj_function, prior, bounds, radius=10e-5):
    """
        Performs Random Search for `obj_function`. Candidates are 
        randomly choosen, however a distance check is always performed 
        in order to avoid repetitions. `radius` parameter controls
        this behaviour, and can be nulled to suppress this check.
    """
    iterations = 10

    progress = {
        "config": np.empty((0, prior["X"].shape[1])),
        "value": np.empty((0, 1)),
        "time": np.empty((0, 1))
    }

    wallclock_time = time.time()

    for _ in range(iterations):
        logging.info(f"---- RS: Iteration #{i+1} ----")
        
        X_candidate = get_candidate(prior, bounds)

        logging.info(f"Selected this candidate: {X_candidate}")

        logging.info("Evaluating function...")

        y_candidate = obj_function(X_candidate)
        iteration_time = time.time() - wallclock_time

        logging.info(f"Function value: {y_candidate}")

        prior["X"] = np.vstack([prior["X"], X_candidate])
        prior["y"] = np.append(prior["y"], np.array([y_candidate]))

        progress["config"] = np.vstack([progress["config"], X_candidate])
        progress["value"] = np.append(progress["value"], np.array([y_candidate]))
        progress["time"] = np.append(progress["time"], np.array([iteration_time]))

    prior["y_best"] = max(prior["y"])
    imax = np.argmax(prior["y"])
    prior["X_best"] = prior["X"][imax]

    return prior, progress


def get_candidate(prior, bounds, radius=10e-5):

    def compute_distance(candidate, points):
        return list(
            map(
                lambda x: np.linalg.norm(x - candidate), 
                points
            )
        )
    x_candidate = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(1, prior["X"].shape[1])
        )

    while min(compute_distance(x_candidate, prior["X"])) < radius:
        x_candidate = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(1, prior["X"].shape[1])
        )

    return x_candidate
