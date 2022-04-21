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


def perceptron_accuracy(X, y, weights, offset_enabled=False):
    """
    acc = 1/n * sum(y_i != sign(w^T x_i))
    """
    X_ = X
    if offset_enabled:
        X_ = np.hstack((np.ones((X.shape[0], 1)), X))
    acc = ((y * (weights @ X_.T)) > 0).mean()
    return acc


def perceptron_algorithm(X, y, offset_enabled=False, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    X_ = X
    if offset_enabled:
        X_ = np.hstack((np.ones((X.shape[0], 1)), X)) # [1, x_1, x_2, ..., x_m]
    n, m = X_.shape
    num_iter = 0
    weights = np.zeros(m)
    acc = 0
    while acc < 1 and num_iter < max_iter:
        for i in range(n):
            if y[i] * (weights @ X_[i].T) <= 0:
                weights = weights + y[i] * X_[i]
                num_iter += 1
        acc = perceptron_accuracy(X, y, weights, offset_enabled)
        # print(f"iter: {num_iter}, acc: {acc}")
    return weights


def evaluate(filename, features, y_col, offset_enabled=False, max_iter=1000):
    print("==========================")
    X_train, y_train, X_test, y_test = prepare_data(filename, features, y_col)
    weights = perceptron_algorithm(X_train, y_train, offset_enabled, max_iter)
    train_acc = perceptron_accuracy(X_train, y_train, weights, offset_enabled)
    test_acc = perceptron_accuracy(X_test, y_test, weights, offset_enabled)
    print(f"Model: {y_col} = {'w_0 + ' if offset_enabled else ''}{' + '.join([f'w_{i+1} * {x_cols[i]}' for i in range(len(x_cols))])}")
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print("==========================")
    return train_acc, test_acc



if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    print("Perceptron w/ Offset")
    for x_cols in features:
        evaluate("Data/binary_classification.csv", x_cols, y_col, True, 1000)
    print("Perceptron w/o Offset")
    for x_cols in features:
        evaluate("Data/binary_classification.csv", x_cols, y_col, False, 1000)

