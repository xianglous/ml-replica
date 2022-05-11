import sys
sys.path.append('..')
import numpy as np
import time
from utils.data import Dataset, model_str
from utils.algorithm import SGD, GD
from utils.loss import logistic_loss
from utils.function import sigmoid
from utils.metrics import accuracy


def logistic_GD_grad(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return X.T @ (sigmoid(X, weights) - y)


def logistic_SGD_grad(X, y, weights):
    """
    X: (m, )
    y: scalar
    weights: (m, )
    """
    return X * (sigmoid(X, weights) - y)


def logistic_regression(X, y, regularization=None, reg=1.0, method='SGD', lr=1e-3, tol=1e-3, max_iter=1000):
    if method == 'SGD':
        return SGD(X, y, logistic_SGD_grad, logistic_loss, regularization, reg, lr, tol, max_iter)
    elif method == 'GD':
        return GD(X, y, logistic_GD_grad, logistic_loss, regularization, reg, lr, tol, max_iter)
    else:
        raise ValueError("method must be 'SGD', 'GD', or 'closed_form'")


def evaluate(data, x_cols, y_col, regularization, reg, method, lr=1e-3, tol=1e-3, max_iter=1000):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    weights = logistic_regression(X_train, y_train, regularization, reg, method, lr, tol, max_iter)
    train_loss = logistic_loss(X_train, y_train, weights)
    test_loss = logistic_loss(X_test, y_test, weights)
    ty_train_pred = sigmoid(X_train, weights) > 0.5
    ty_test_pred = sigmoid(X_test, weights) > 0.5
    train_acc = accuracy(y_train, ty_train_pred)
    test_acc = accuracy(y_test, ty_test_pred)
    print(model_str(x_cols, y_col, True, 'logistic')+f" using {method} logistic regression with {regularization} regulatization")
    print(f"Train loss: {train_loss}")
    print(f"Test loss: {test_loss}")
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    data = Dataset("../data/binary_classification.csv")
    data.transform(["age", "interest"], "standardize")
    for method in ["SGD", "GD"]:
        for x_cols in features:
            for regularization in [None, "l1", "l2"]:
                evaluate(data, x_cols, y_col, regularization, 1.0, method)