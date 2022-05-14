import sys
sys.path.append('..')
import time
from mlreplica.linear_model import LogisticRegression
from mlreplica.utils.data import Dataset
from mlreplica.utils.loss import LogLoss
from mlreplica.utils.metrics import accuracy


def evaluate(data, x_cols, y_col, regularization, reg, lr=1e-3, tol=1e-3, max_iter=1000):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    clf = LogisticRegression(regularization=regularization, alpha=reg, lr=lr, tol=tol, max_iter=max_iter)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    log_loss = LogLoss()
    train_loss = log_loss(y_train, y_train_pred)
    test_loss = log_loss(y_test, y_test_pred)
    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)
    print(clf.str(x_cols, y_col)+f" logistic regression with {regularization} regulatization")
    print(f"Train loss: {train_loss}")
    print(f"Test loss: {test_loss}")
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    data = Dataset("../data/binary_classification.csv")
    data.transform(["age", "interest"], "standardize")
    for x_cols in features:
        for regularization in [None, "l1", "l2"]:
            evaluate(data, x_cols, y_col, regularization, 1.0)