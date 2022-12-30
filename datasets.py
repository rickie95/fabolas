from random import sample

import math
import logging


def load_mnist(training_size=1):
    """ Loads MNIST dataset. It is possible to reduce the training set size
        passing the fraction requested. (e.g. passing 64 il will be loaded
        a 1/64 of the training set). Test set is always fully populated.

        ## Parameters
        - `training_size` Must be equal or greater than 1.
    """

    import mnist
    mnist.temporary_dir = lambda: './data/mnist'

    if training_size > 1:
        logging.error("Training set size must be greater than 1")
        raise ValueError("Training set size must be greater than 1")

    dataset = {}

    train_x, train_y = mnist.train_images(), mnist.train_labels()
    test_x, test_y = mnist.test_images(), mnist.test_labels()

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    if training_size < 1:
        indices = sample(range(0, train_x.shape[0]), math.floor(train_x.shape[0]*training_size))
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


def load_svhn():
    """
        Loads the Street View House Numbers dataset.
    """
    from scipy.io import loadmat
    from sklearn.preprocessing import LabelBinarizer
    import numpy as np
    from random import shuffle

    train_raw = loadmat('./data/svhn/train_32x32.mat')
    test_raw = loadmat('./data/svhn/test_32x32.mat')

    train_images = np.array(train_raw['X'])
    test_images = np.array(test_raw['X'])

    train_labels = train_raw['y']
    test_labels = test_raw['y']
    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)

    train_images = train_images.astype('float64') / 255.0
    test_images = test_images.astype('float64') / 255.0

    train_labels = train_labels.astype('int64')
    test_labels = test_labels.astype('int64')

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    # 73257 images
    # 6000 of them used as validation set

    inds = [x for x in range(train_images.shape[0])]
    shuffle(inds)

    train_ind = inds[0:67257]
    valid_ind = inds[67257:-1]

    dataset = {
        'X' : train_images[train_ind],
        'y' : train_labels[train_ind],
        'X_test' : train_images[valid_ind],
        'y_test' : train_labels[valid_ind]
    }

    del train_raw
    del test_raw
    del train_images
    del train_labels
    del train_ind
    del valid_ind

    return dataset
