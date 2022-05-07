import numpy as np

def accuracy(y_true, y_pred):
    """
    y_true: (n, )
    y_pred: (n, )
    """
    return np.mean(y_true == y_pred)