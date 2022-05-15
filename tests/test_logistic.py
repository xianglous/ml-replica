import sys
sys.path.append('..')
import time
from mlreplica.linear_model import LogisticRegression
from mlreplica.utils.data import Dataset
from mlreplica.utils.metrics import accuracy


def evaluate(data, x_cols, y_col, regularization, reg, lr=1e-3, tol=1e-3, max_iter=1000):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    clf = LogisticRegression(regularization=regularization, alpha=reg, lr=lr, tol=tol, max_iter=max_iter)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(clf.str(x_cols, y_col)+f" logistic regression with {regularization} regulatization")
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