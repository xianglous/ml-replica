import sys
sys.path.append('..')
import numpy as np
from utils.data import prepare_data, model_str
from utils.metrics import accuracy

def perceptron_accuracy(X, y, weights):
    """
    acc = 1/n * sum(y_i != sign(w^T x_i))
    """
    return accuracy(y, np.sign(weights @ X.T))


def perceptron(X, y, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    n, m = X.shape
    num_iter = 0
    weights = np.zeros(m)
    acc = 0
    while acc < 1 and num_iter < max_iter:
        for i in range(n):
            if y[i] * (weights @ X[i].T) <= 0:
                weights = weights + y[i] * X[i]
                num_iter += 1
        acc = perceptron_accuracy(X, y, weights)
    return weights


def evaluate(filename, x_cols, y_col, offset_enabled=False, max_iter=1000):
    print("==========================")
    X_train, y_train, X_test, y_test = prepare_data(filename, x_cols, y_col)
    if offset_enabled:
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    weights = perceptron(X_train, y_train, max_iter)
    train_acc = perceptron_accuracy(X_train, y_train, weights)
    test_acc = perceptron_accuracy(X_test, y_test, weights)
    print(model_str(x_cols, y_col, offset_enabled))
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print("==========================")
    return train_acc, test_acc


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    print("Perceptron w/ Offset")
    for x_cols in features:
        evaluate("../Data/binary_classification.csv", x_cols, y_col, True, 1000)
    print("Perceptron w/o Offset")
    for x_cols in features:
        evaluate("../Data/binary_classification.csv", x_cols, y_col, False, 1000)

