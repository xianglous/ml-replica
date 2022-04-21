import numpy as np
import pandas as pd


def prepare_data(filename, x_cols, y_col, ratio=0.8, random_state=42):
    df = pd.read_csv(filename)
    df[y_col] = df[y_col] * 2 - 1 # -1 or 1
    train_df = df.sample(frac=ratio, random_state=random_state)
    test_df = df.drop(train_df.index)
    X_train, y_train = train_df[x_cols].to_numpy(), train_df[y_col].to_numpy()
    X_test, y_test = test_df[x_cols].to_numpy(), test_df[y_col].to_numpy()
    return X_train, y_train, X_test, y_test


def perceptron_accuracy(X, y, weights):
    """
    acc = 1/n * sum(y_i != sign(w^T x_i))
    """
    acc = ((y * (weights @ X.T)) > 0).mean()
    return acc


def perceptron_algorithm(X, y, max_iter=10):
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
        acc = perceptron_accuracy(X, y, weights)
        # print(f"iter: {num_iter}, acc: {acc}")
        num_iter += 1
    return weights


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    for x_cols in features:
        X_train, y_train, X_test, y_test = prepare_data(
            "Data/binary_classification.csv",
            x_cols, y_col)
        weights = perceptron_algorithm(X_train, y_train)
        train_acc = perceptron_accuracy(X_train, y_train, weights)
        test_acc = perceptron_accuracy(X_test, y_test, weights)
        print("==========================")
        print(f"Model: {y_col} = {' + '.join([f'w_{i+1} * {x_cols[i]}' for i in range(len(x_cols))])}")
        print(f"Train accuracy: {train_acc}")
        print(f"Test accuracy: {test_acc}")
