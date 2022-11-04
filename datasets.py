from random import sample

import math
import logging

import mnist


def load_mnist(training_size=1):
    import mnist
    """ Loads MNIST dataset. It is possible to reduce the training set size
        passing the fraction requested. (e.g. passing 64 il will be loaded
        a 1/64 of the training set). Test set is always fully populated.

        ## Parameters
        - `training_size` Must be equal or greater than 1.
    """

    if training_size > 1:
        logging.error("Training set size must be greater than 1")
        raise ValueError("Training set size must be greater than 1")

    dataset = {}

    train_x, train_y = mnist.train_images(), mnist.train_labels()
    test_x, test_y = mnist.test_images(), mnist.test_labels()

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    if training_size < 1:
        indices = sample(range(1, train_x.shape[0]), math.floor(train_x.shape[0]*training_size))
        dataset["X"] = train_x[indices]
        dataset["y"] = train_y[indices]
    else:
        dataset["X"] = train_x
        dataset["y"] = train_y

    dataset["X_test"] = test_x
    dataset["y_test"] = test_y

    return dataset


def load_cifar(train_size=1):
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical

    """
    Load CIFAR10 dataset from keras. Image pixes vary from 0 to 255, so a normalization is needed.
    One hot encoding is used for the categories (10 in total).
    """

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    X_train = X_train/255
    Y_train_en = to_categorical(Y_train, 10)

    dataset = {}

    if train_size < 1:
        indices = sample(
            range(1, X_train.shape[0]),
            math.floor(X_train.shape[0] * train_size))

        dataset["X"] = X_train[indices]
        dataset["y"] = Y_train_en[indices]
    else:
        dataset["X"] = X_train
        dataset["y"] = Y_train_en

    dataset["X_test"] = X_test/255
    dataset["y_test"] = to_categorical(Y_test, 10)

    return dataset
