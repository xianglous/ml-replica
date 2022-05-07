import sys
sys.path.append('..')
import numpy as np
import time
from utils.data import prepare_data, model_str
from utils.preprocessing import standardizer, min_max_scaler
from utils.algorithm import SGD, GD
from utils.loss import MSE_loss


def lasso_SGD_grad(X, y, weights):
    return -X * (y - X @ weights)


def lasso_GD_grad(X, y, weights):
    return -X.T @ (y - X @ weights) / X.shape[0]


def lasso_regression(X, y, reg=1.0, method='SGD', lr=1e-3, tol=1e-3, max_iter=1000):
    if method == 'SGD':
        return SGD(X, y, lasso_SGD_grad, MSE_loss, 'l1', reg, lr, tol, max_iter)
    elif method == 'GD':
        return GD(X, y, lasso_GD_grad, MSE_loss, 'l1', reg, lr, tol, max_iter)
    else:
        raise ValueError("method must be 'SGD' or 'GD'")


def evaluate(filename, x_cols, y_col, reg, method, transform=None, lr=1e-5, tol=1e-3, max_iter=1000):
    print("==========================")
    start = time.time()
    
    X_train, y_train, X_test, y_test = prepare_data(filename, x_cols, y_col, X_transform=transform, y_transform=transform)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    weights = lasso_regression(X_train, y_train, reg, method, lr, tol, max_iter)
    train_loss = MSE_loss(X_train, y_train, weights)
    test_loss = MSE_loss(X_test, y_test, weights)
    print(model_str(x_cols, y_col, True)+f" using {method} lasso regression with reg={reg} and {transform} normalization")
    print(f"Train loss: {train_loss}")
    print(f"Test loss: {test_loss}")
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == "__main__":
    features = [["bedrooms", "bathrooms", "sqft_living", "floors"]]
    y_col = "price"
    for method in ["SGD", "GD"]:
        for x_cols in features:
            for normalization in ["standard", "min_max"]:
                evaluate("../Data/regression_data.csv", x_cols, y_col, 1.0, method, normalization)