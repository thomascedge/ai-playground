import numpy as np

class LinearRegression:
    def __init__(self, lr:int=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains data based on gradient descent function: y_pred = wX + b where 
        y-hat is the estimated value, w is the weight associated with a feature
        influencing the slope, X is the series of inputs, and b is the bias
        influencing the fitness of a line to fit the data.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias    # dot product in np includes summation
            
            # updates weights and biases after predictoin
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weight = self.weight = self.lr * dw
            self.bias = self.bias = self.lr * db

    def predict(self, X):
        """
        Applies equation made during training to testing dataset
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    