import sys

from sklearn.ensemble import BaggingClassifier
sys.path.append('..')
import numpy as np
from mlreplica.utils.data import Dataset
from mlreplica.utils.algorithm import GD, SGD
from mlreplica.linear_model import LinearModel
from mlreplica.utils.loss import HingeLoss
from mlreplica.utils.metrics import accuracy
from mlreplica.utils.preprocessing import perceptronizer


def evaluate(data, x_cols, y_col, method='SGD', lr=1e-3, tol=1e-3, max_iter=1000):
    print("==========================")
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    clf = LinearModel(
        bias=True,
        solver=SGD if method == 'SGD' else GD, 
        transform=np.sign, 
        loss=HingeLoss(), 
        lr=lr, 
        tol=tol, 
        max_iter=max_iter)
    clf.fit(X_train, y_train)
    train_acc = accuracy(y_train, clf.predict(X_train))
    test_acc = accuracy(y_test, clf.predict(X_test))
    print(clf.str(x_cols, y_col)+f" using {method}")
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print("==========================")
    return train_acc, test_acc


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    print("Gradient Descent")
    data = Dataset("../data/binary_classification.csv")
    # data.transform(["age", "interest"], "standardize")
    data.transform("success", perceptronizer)
    for method in ['GD', 'SGD']:
        for x_cols in features:
            evaluate(data, x_cols, y_col, method)