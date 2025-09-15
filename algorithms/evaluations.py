import numpy as np

def mse(y_test, predictions):
    "Mean squared error"
    return np.mean((y_test - predictions) ** 2)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)
