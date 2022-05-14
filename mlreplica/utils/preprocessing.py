import numpy as np


class MinMaxScaler:
    """Scales the data to the range [0, 1]"""
    def __init__(self, X, y=None):
        if X is None:
            X = np.array([y]).T
            y = None
        if y is not None:
            X = np.column_stack((X, y))
        self.max = X.max(axis=0)
        self.min = X.min(axis=0)
        same = self.max == self.min
        self.max[same] = self.max[same] + 1e-6
    
    def __call__(self, X, y=None):
        if y is not None:
            X = np.column_stack((X, y))
        normalized = (X - self.min) / (self.max - self.min)
        if y is not None:
            return normalized[:, :-1], normalized[:, -1]
        return normalized


class Standardizer:
    """Standardize the data using the mean and standard deviation"""
    def __init__(self, X, y=None):
        if X is None:
            X = np.array([y]).T
            y = None
        if y is not None:
            X = np.column_stack((X, y))
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        same = self.std == 0
        self.std[same] = 1e-6
        self.mean[same] -= 1e-6
    
    def __call__(self, X, y=None):
        if y is not None:
            X = np.column_stack((X, y))
        standardized = (X - self.mean) / self.std
        if y is not None:
            return standardized[:, :-1], standardized[:, -1]
        return standardized


def perceptronizer(y, pos_class=1):
    """
    y: (n, )
    """
    y = y.copy()
    y[y != pos_class] = -1
    y[y == pos_class] = 1
    return y

