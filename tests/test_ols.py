import sys
sys.path.append('..')
import numpy as np
import time
from mlreplica.linear_model import LinearRegression
from mlreplica.utils.data import Dataset


def evaluate(data, x_cols, y_col, method, lr=1e-2, tol=1e-3, max_iter=1000):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    model = LinearRegression(regularization=None, solver=method, lr=lr, tol=tol, max_iter=max_iter)
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(model.str(x_cols, y_col)+f" using {method} OLS")
    print(f"Train R2: {train_r2}")
    print(f"Test R2: {test_r2}")
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