import sys
sys.path.append('..')
import time
from mlreplica.utils.data import Dataset
from mlreplica.linear_model import LassoRegression


def evaluate(data, x_cols, y_col, reg, method, lr=1e-2, tol=1e-3, max_iter=1000):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    model = LassoRegression(solver=method, alpha=reg, lr=lr, tol=tol, max_iter=max_iter)
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(model.str(x_cols, y_col)+f" using {method} lasso regression with reg={reg}")
    print(f"Train R2: {train_r2}")
    print(f"Test R2: {test_r2}")
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == "__main__":
    features = [["bedrooms", "bathrooms", "sqft_living", "floors"]]
    y_col = "price"
    data = Dataset("../Data/regression_data.csv", random_state=42)
    data.transform(["bedrooms", "bathrooms", "sqft_living", "floors", "price"], "standardize")
    for method in ["SGD", "GD"]:
        for x_cols in features:
            evaluate(data, x_cols, y_col, 0, method)