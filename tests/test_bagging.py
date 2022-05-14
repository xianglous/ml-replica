import sys 
sys.path.append('..')
import time
from mlreplica.utils.data import Dataset
from mlreplica.utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
from mlreplica.ensemble_model import BaggingClassifier
from mlreplica.tree_model import DecisionTree


def evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy", random_state=None):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    base = DecisionTree(criterion=criterion, max_depth=max_depth)
    clf = BaggingClassifier(
        base_estimator=base, 
        n_estimators=5, 
        max_samples=0.8, 
        max_features=0.4, 
        n_jobs=3,
        random_state=random_state)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("Training accuracy:", accuracy(y_train, y_train_pred))
    print("Testing accuracy:", accuracy(y_test, y_test_pred))
    print("Training Confusion matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    print("Testing Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == '__main__':
    data = Dataset("../Data/drug200.csv", random_state=42)
    x_cols = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
    y_col = "Drug"
    evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy", random_state=42)