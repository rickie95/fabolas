"""
    Performs experiments on a CNN trained on CIFAR10
"""

import logging
import sys
import datetime

import numpy as np
import pandas as pd

from datasets import load_cifar
from utils import save_results, print_usage

from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization


def obj_function(configuration):
    """
    Model building, the CNN has the following properties:
    * 3 convolutional layers, each one with batch normalization
    * optimized using Adam
    * a dense layer

    Returns the validation error and the time spent.
    """

    training_set_size = 1

    assert 4 < len(
        configuration) < 7, "Configuration must hold 5 parameters (6 if running FABOLAS)"

    l1_filters, l2_filters, l3_filters, batch_size, learning_rate = configuration[0:5]

    if len(configuration) == 6:
        training_set_size = configuration[-1]

    dataset = load_cifar(training_set_size)

    # Remap log scale parameters
    l1_filters = (int) (2**np.round(l1_filters))
    l2_filters = (int) (2**np.round(l2_filters))
    l3_filters = (int) (2**np.round(l3_filters))

    batch_size = (int) (np.round(batch_size))

    learning_rate = 10**learning_rate

    model = Sequential()

    # Layer 1
    model.add(Conv2D(l1_filters, (5, 5), input_shape=(
        32, 32, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same"))

    # Layer 2
    model.add(Conv2D(l2_filters, (5, 5), input_shape=(
        32, 32, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same"))

    # Layer 3
    model.add(Conv2D(l3_filters, (5, 5), input_shape=(
        32, 32, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.summary()
    history = model.fit(
        dataset["X"], 
        dataset["y"], 
        epochs=40, 
        batch_size=batch_size,
        verbose=1,
        validation_data=(dataset["X_test"], dataset["y_test"])
        )

    val_accuracy = history.history['val_accuracy'][-1]

    return val_accuracy


def generate_prior(bounds, n_points=25):
    prior = {
        "X": np.empty((0, 2)),
        "y": np.empty((0, 1)),
        "s": np.empty((0, 1)),
        "c": np.empty((0, 1))
    }

    prior_candidates = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_points, len(bounds)))

    for configuration in prior_candidates:
        y, c = obj_function(configuration)
        prior["X"] = np.vstack([prior["X"], np.array(configuration)])
        prior["y"] = np.append(prior["y"], np.array([y]))
        prior["s"] = np.append(prior["s"], np.array([y]))
        prior["c"] = np.append(prior["c"], np.array([y]))


def load_prior(with_size=1):
    df = pd.read_csv("./prior/prior_cnn_cifar10.csv")

    if not with_size:
        df = df.query("size == 1")

    return {
        "X": np.concatenate(
            (np.log2(
                np.array([df["l1_filters"],
                          df["l2_filters"],
                          df["l3_filters"]])
            ),
                np.array([df["batch_size"]]),
                np.log10(np.array([df["learn_rate"]]))),
            axis=0).T,
        "y": np.array(df["score"]).reshape(-1, 1),
        "c": np.array(df["cost_s"]).reshape(-1, 1),
        "size": np.array(df["size"]).reshape(-1, 1)
    }


def cnn_cifar10(method='random_search', save_path=None):
    save_path = "./results/cnn_cifar10" if save_path is None else save_path
    print(f"Log will be found at {save_path}cnn_cifar10_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log")

    logging.basicConfig(
        format='CNN_CIFAR10 (%(process)s) - %(levelname)s - %(message)s',
        level=logging.INFO,
        filename=f"{save_path}cnn_cifar10_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log",
        filemode="w",
        force=True
    )

    prior = None

    method = 'random_search' if method is None else method

    try:
        prior = load_prior()
    except:
        prior = generate_prior()

    assert (prior is not None)

    # Dry run for caching data
    load_cifar()

    bounds = [
        (4, 9),
        (4, 9),
        (4, 9),
        (32, 512),
        (-6, -1)
    ]

    results = []

    if method == 'random_search':
        from random_search import rs
        logging.info("Starting Random Search...")
        results, progress = rs(obj_function, prior, bounds)

    elif method == 'expected_improvement':
        from expected_improvement import ei
        logging.info("Starting Expected Improvement...")
        results, progress = ei(obj_function, prior, bounds)

    elif method == 'entropy_search':
        from entropy_search import es
        logging.info("Starting Entropy Search...")
        results, progress = es(obj_function, prior, bounds)

    elif method == 'fabolas':
        from fabolas import fabolas
        logging.info("Starting FABOLAS...")
        bounds.append((1/256, 1))
        results, progress = fabolas(obj_function, load_prior(
            with_size=True), bounds)

    else:
        return 1

    logging.info(f"Best value: {prior['y_best']}, with conf {prior['X_best']}")

    save_results(
        results,
        progress,
        method,
        save_path
    )

    return 0


if __name__ == "__main__":
    method = None
    save_path = None

    print(f"Passed {len(sys.argv)} arguments:")

    if len(sys.argv) > 1:
        method = sys.argv[1]
        print(f"method={sys.argv[1]}")

    if len(sys.argv) > 2:
        save_path = sys.argv[2]
        print(f"save_path={sys.argv[2]}")

    if cnn_cifar10(method=method, save_path=save_path) > 0:
        print_usage(sys.argv)
