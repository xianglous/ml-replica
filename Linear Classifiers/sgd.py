import numpy as np
from utils import prepare_data, model_str
from perceptron import perceptron_accuracy


def gradient_descent(X, y, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    loss, prev_loss = None, None
    while num_iter < max_iter:
        conds = (y * (weights @ X.T) < 1) # condition check
        weights = weights + lr * (conds * X.T) @ (conds * y) / n
        prev_loss = loss
        loss = np.sum(np.maximum(y * (weights @ X.T), 0))
        if prev_loss and abs(loss-prev_loss) < tol:
            return weights
        num_iter += 1
    return weights


def stochastic_gradient_descent(X, y, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    loss, prev_loss = None, None
    while num_iter < max_iter:
        for i in range(n):
            if y[i] * (weights @ X[i].T) < 1:
                weights = weights + lr * y[i] * X[i]
                prev_loss = loss
                loss = np.sum(np.maximum(y * (weights @ X.T), 0))
                if prev_loss and abs(loss-prev_loss) < tol:
                    return weights
                num_iter += 1
    return weights


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
        evaluate("Data/binary_classification.csv", x_cols, y_col, False)
    print("Stochastic Gradient Descent")
    for x_cols in features:
        evaluate("Data/binary_classification.csv", x_cols, y_col, True)