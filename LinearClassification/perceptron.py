import sys
sys.path.append('..')
import numpy as np
from utils.data import Dataset, model_str
from utils.metrics import accuracy
from utils.preprocessing import perceptronizer


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
        acc = accuracy(y, np.sign(weights @ X.T))
    return weights


def evaluate(data, x_cols, y_col, offset_enabled=False, max_iter=1000):
    print("==========================")
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    if offset_enabled:
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    weights = perceptron(X_train, y_train, max_iter)
    y_train_pred = np.sign(weights @ X_train.T)
    y_test_pred = np.sign(weights @ X_test.T)
    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)
    print(model_str(x_cols, y_col, offset_enabled))
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print("==========================")
    return train_acc, test_acc


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    print("Perceptron w/ Offset")
    data = Dataset("../data/binary_classification.csv")
    # data.transform(["age", "interest"], "standardize")
    data.transform("success", perceptronizer)
    for x_cols in features:
        evaluate(data, x_cols, y_col, True, 1000)
    print("Perceptron w/o Offset")
    for x_cols in features:
        evaluate(data, x_cols, y_col, False, 1000)

