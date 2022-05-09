import sys
sys.path.append('..')
import numpy as np
import time
from utils.data import dataset, model_str
from utils.algorithm import SGD, GD
from utils.loss import MSE_loss


def ridge_closed_form(X, y, reg):
    return np.linalg.pinv(X.T @ X + reg * np.eye(X.shape[1])) @ X.T @ y


def ridge_SGD_grad(X, y, weights):
    return -X * (y - X @ weights)


def ridge_GD_grad(X, y, weights):
    return -X.T @ (y - X @ weights) / X.shape[0]


def ridge_regression(X, y, reg=1.0, method='SGD', lr=1e-3, tol=1e-3, max_iter=1000):
    if method == 'SGD':
        return SGD(X, y, ridge_SGD_grad, MSE_loss, 'l2', reg, lr, tol, max_iter)
    elif method == 'GD':
        return GD(X, y, ridge_GD_grad, MSE_loss, 'l2', reg, lr, tol, max_iter)
    elif method == 'closed_form':
        return ridge_closed_form(X, y, reg)
    else:
        raise ValueError("method must be 'SGD', 'GD', or 'closed_form'")


def evaluate(data, x_cols, y_col, reg, method, lr=1e-2, tol=1e-3, max_iter=1000):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    weights = ridge_regression(X_train, y_train, reg, method, lr, tol, max_iter)
    train_loss = MSE_loss(X_train, y_train, weights)
    test_loss = MSE_loss(X_test, y_test, weights)
    print(model_str(x_cols, y_col, True)+f" using {method} ridge regression with reg={reg}")
    print(f"Train loss: {train_loss}")
    print(f"Test loss: {test_loss}")
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == "__main__":
    features = [["bedrooms", "bathrooms", "sqft_living", "floors"]]
    y_col = "price"
    data = dataset("../Data/regression_data.csv", random_state=42)
    data.transform(["bedrooms", "bathrooms", "sqft_living", "floors", "price"], "standardize")
    for method in ["SGD", "GD", "closed_form"]:
        for x_cols in features:
            evaluate(data, x_cols, y_col, 1.0, method)