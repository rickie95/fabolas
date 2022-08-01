import random
from math import floor

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.datasets import mnist

training_size = 50
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
indices = random.sample(range(1, train_x.shape[0]), floor(train_x.shape[0]/training_size))

params = {"C":[0.1, 1, 10], "gamma": [0.1, 0.01, 0.001]}
svc = svm.SVC(kernel='rbf')
grid_search = GridSearchCV(svc, params)
grid_search.fit(train_x[indices], train_y[indices])
print(f"Best score: {grid_search.best_score_}")

