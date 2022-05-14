import sys
sys.path.append('..')
from mlreplica.linear_model import SVM
from mlreplica.utils.data import Dataset
from mlreplica.utils.metrics import accuracy, precision, recall, f1_score
from mlreplica.utils.preprocessing import perceptronizer


def evaluate(data, x_cols, y_col, C, kernel, tol, heuristic, max_iter):
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    print("=========RANDOM==========" if not heuristic else "=========HEURISTIC==========")
    print("Kernel:", kernel)
    clf = SVM(C=C, kernel=kernel, tol=tol, heuristic=heuristic, max_iter=max_iter)
    if kernel == 'precomputed':
        X_test = X_test@X_train.T
        X_train = X_train@X_train.T
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy(y_test, y_pred))
    print("Precision:", precision(y_test, y_pred))
    print("Recall:", recall(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("=========================")


if __name__ == "__main__":
    x_cols = ["age", "interest"]
    y_col = "success"
    data = Dataset("../data/binary_classification.csv")
    data.transform(["age", "interest"], "standardize")
    data.transform("success", perceptronizer)
    # for kernel in ['linear', 'poly', 'rbf', 'precomputed']:
    #      evaluate(data, x_cols, y_col, 1, kernel, 0.1, False, 10)
    for kernel in ['linear', 'poly', 'rbf', 'precomputed']:
         evaluate(data, x_cols, y_col, 1, kernel, 0.1, False, 3)