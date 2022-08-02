import os, random
from math import floor, log10

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.datasets import mnist


base_path = "./results/svm-grid-search"

def perform_experiment(train_x, train_y, test_x, test_y, training_size, C_values, gamma_values):
    print(f"\n=== STARTING EXPERIMENT WITH 1/{training_size} DATA ===\n")

    # Extract a subset of training data, uniformly chosen
    indices = random.sample(range(1, train_x.shape[0]), floor(train_x.shape[0]/training_size))

    # Create the grid search object
    grid_search = GridSearchCV(svm.SVC(kernel="rbf"), {'C': C_values, 'gamma' : gamma_values}, n_jobs=-1, verbose=3, cv=3)

    # Fit the models, using only the examples selected by the indices list
    grid_search.fit(train_x[indices], train_y[indices])

    predictions = grid_search.predict(test_x)

    print("\n===== EXPERIMENT SUMMARY ======\n")
    print(f"Best score {grid_search.best_score_} with parameters {grid_search.best_params_}")
    print(classification_report(test_y, predictions))

    # Assemble a list of tuples (parameters, mean_test_score)
    search_results = zip(grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"])

    # Then convert it into a matrix, in order to plot and dump as a csv
    results_matrix = np.zeros((len(C_values), len(gamma_values)))
    for r in search_results:
        results_matrix[C_values.index(r[0]["C"])][gamma_values.index(r[0]["gamma"])] = r[1]

    np.savetxt(f"{base_path}/results-{training_size}.csv", results_matrix, delimiter=",")

    # Plot the experiment results
    plt.xticks(ticks=np.arange(len(gamma_values)),labels=[log10(x) for x in gamma_values])
    plt.yticks(ticks=np.arange(len(C_values)),labels=[log10(x) for x in C_values])
    plt.xlabel('log(Gamma)')
    plt.ylabel('log(C)')
    hm = plt.imshow(results_matrix, interpolation='spline16', cmap="Blues")
    plt.colorbar(hm)
    plt.savefig(f"{base_path}/results-{training_size}.png")
    #plt.show()
    print("\n===== END EXPERIMENT ======\n")

def svm_grid_search(verbosity=1):
    print("=============================================================")
    print("===== SVM Hyperparameter Optimization with Grid Search ======")
    print("=============================================================")

    training_set_size = [128, 16, 4, 1]

    if not os.path.exists(base_path):
            os.makedirs(base_path)

    print("Loading MNIST dataset...")
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    # Generate a list of potential hyperparameters candidates
    C_values = [10**(x) for x in range(-10, 10)]
    gamma_values = [10**(x) for x in range(-10, 10)]

    for training_size in training_set_size:
        perform_experiment(train_x, train_y, test_x, test_y, training_size, C_values, gamma_values)

if __name__ == "__main__":
    svm_grid_search()
