import sys
sys.path.append('..')
import numpy as np
import time
from utils.data import Dataset, model_str
from utils.algorithm import SGD, GD
from utils.loss import MSE_loss


def ols_closed_form(X, y):
    """
    X: (n, m)
    y: (n, )
    w = (X.T @ X)^-1 @ X.T @ y
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def MSE_SGD_grad(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    grad = -x * (y - x @ weights)
    """
    return -X * (y - X @ weights)


def MSE_GD_grad(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    grad = -1 / n * sum_i(xi * (yi - xi @ weights))
    """
    return -X.T @ (y - X @ weights) / X.shape[0]


def ols(X, y, method='SGD', lr=1e-3, tol=1e-3, max_iter=1000):
    if method == 'SGD':
        return SGD(X, y, MSE_SGD_grad, MSE_loss, lr=lr, tol=tol, max_iter=max_iter)
    elif method == 'GD':
        return GD(X, y, MSE_GD_grad, MSE_loss, lr=lr, tol=tol, max_iter=max_iter)
    elif method == 'closed-form':
        return ols_closed_form(X, y)
    else:
        raise ValueError("method must be 'SGD', 'GD', or 'closed-form'")


def evaluate(data, x_cols, y_col, method, lr=1e-2, tol=1e-3, max_iter=1000):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    weights = ols(X_train, y_train, method, lr, tol, max_iter)
    train_loss = MSE_loss(X_train, y_train, weights)
    test_loss = MSE_loss(X_test, y_test, weights)
    print(model_str(x_cols, y_col, True)+f" using {method} OLS")
    print(f"Train loss: {train_loss}")
    print(f"Test loss: {test_loss}")
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == "__main__":
    features = [["bedrooms", "bathrooms", "sqft_living", "floors"]]
    y_col = "price"
    data = Dataset("../Data/regression_data.csv", random_state=42)
    data.transform(["bedrooms", "bathrooms", "sqft_living", "floors", "price"], "standardize")
    for method in ["SGD", "GD", "closed-form"]:
        for x_cols in features:
            evaluate(data, x_cols, y_col, method)