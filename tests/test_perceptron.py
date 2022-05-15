import sys
sys.path.append('..')
from mlreplica.linear_model import Perceptron
from mlreplica.utils.data import Dataset
from mlreplica.utils.metrics import accuracy
from mlreplica.utils.preprocessing import perceptronizer


def evaluate(data, x_cols, y_col, offset_enabled=False, max_iter=1000):
    print("==========================")
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    clf = Perceptron(bias=offset_enabled, max_iter=max_iter)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(clf.str(x_cols, y_col))
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print("==========================")
    return train_acc, test_acc


if __name__ == "__main__":
    features = [["age"], ["interest"], ["age", "interest"]]
    y_col = "success"
    print("Perceptron w/ Offset")
    data = Dataset("../data/binary_classification.csv")
    # data.transform(["age", "interest"], "standardize")
    data.transform("success", perceptronizer)
    for x_cols in features:
        evaluate(data, x_cols, y_col, True, 1000)
    print("Perceptron w/o Offset")
    for x_cols in features:
        evaluate(data, x_cols, y_col, False, 1000)

