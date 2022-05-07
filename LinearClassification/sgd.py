import sys
sys.path.append('..')
import numpy as np
from utils.data import prepare_data, model_str
from utils.algorithm import GD, SGD
from utils.loss import hinge_loss
from perceptron import perceptron_accuracy


def hinge_gradient(X, y, weights):
    conds = (y * (weights @ X.T) < 1)
    return -y[conds] @ X[conds] / (1 if len(X.shape)==1 else X.shape[0])


def gradient_descent(X, y, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    return GD(X, y, hinge_gradient, hinge_loss, lr, tol, max_iter)


def stochastic_gradient_descent(X, y, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    return SGD(X, y, hinge_gradient, hinge_loss, lr, tol, max_iter)


def evaluate(filename, x_cols, y_col, stochastic=True, lr=1e-3, tol=1e-3, max_iter=1000):
    print("==========================")
    X_train, y_train, X_test, y_test = prepare_data(filename, x_cols, y_col)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    weights = stochastic_gradient_descent(X_train, y_train, lr, tol, max_iter) if stochastic \
        else gradient_descent(X_train, y_train, lr, tol, max_iter)
    train_acc = perceptron_accuracy(X_train, y_train, weights)
    test_acc = perceptron_accuracy(X_test, y_test, weights)
    print(model_str(x_cols, y_col, True)+f" using {'S' if stochastic else ''}GD")
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print("==========================")
    return train_acc, test_acc


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    print("Gradient Descent")
    for x_cols in features:
        evaluate("../Data/binary_classification.csv", x_cols, y_col, False)
    print("Stochastic Gradient Descent")
    for x_cols in features:
        evaluate("../Data/binary_classification.csv", x_cols, y_col, True)