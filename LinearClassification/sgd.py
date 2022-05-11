import sys
sys.path.append('..')
import numpy as np
from utils.data import Dataset, model_str
from utils.algorithm import GD, SGD
from utils.loss import hinge_loss
from utils.metrics import accuracy
from utils.preprocessing import perceptronizer


def hinge_gradient(X, y, weights):
    conds = (y * (weights @ X.T) < 1)
    return -y[conds] @ X[conds] / (1 if len(X.shape)==1 else X.shape[0])


def gradient_descent(X, y, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    return GD(X, y, hinge_gradient, hinge_loss, lr=lr, tol=tol, max_iter=max_iter)


def stochastic_gradient_descent(X, y, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    return SGD(X, y, hinge_gradient, hinge_loss, lr=lr, tol=tol, max_iter=max_iter)


def evaluate(data, x_cols, y_col, stochastic=True, lr=1e-3, tol=1e-3, max_iter=1000):
    print("==========================")
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    weights = stochastic_gradient_descent(X_train, y_train, lr, tol, max_iter) if stochastic \
        else gradient_descent(X_train, y_train, lr, tol, max_iter)
    y_train_pred = np.sign(weights @ X_train.T)
    y_test_pred = np.sign(weights @ X_test.T)
    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)
    print(model_str(x_cols, y_col, True)+f" using {'S' if stochastic else ''}GD")
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print("==========================")
    return train_acc, test_acc


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    print("Gradient Descent")
    data = Dataset("../data/binary_classification.csv")
    # data.transform(["age", "interest"], "standardize")
    data.transform("success", perceptronizer)
    for x_cols in features:
        evaluate(data, x_cols, y_col, False)
    print("Stochastic Gradient Descent")
    for x_cols in features:
        evaluate(data, x_cols, y_col, True)