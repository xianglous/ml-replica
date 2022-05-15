import sys 
sys.path.append('..')
import time
from mlreplica.utils.data import Dataset
from mlreplica.utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
from mlreplica.tree_model import DecisionTreeClassifier


def evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy", verbose=False):
    if verbose:
        print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    if verbose:
        print("Training accuracy:", train_acc)
        print("Testing accuracy:", test_acc)
        print("Training precision:")
        for cls in sorted(set(y_train)):
            print(f"{data.translate(y_col, cls)}: {precision(y_train, y_train_pred, cls)}")
        print("Testing precision:")
        for cls in sorted(set(y_train)):
            print(f"{data.translate(y_col, cls)}: {precision(y_test, y_test_pred, cls)}")
        print("Training recall:")
        for cls in sorted(set(y_train)):
            print(f"{data.translate(y_col, cls)}: {recall(y_train, y_train_pred, cls)}")
        print("Testing recall:")
        for cls in sorted(set(y_train)):
            print(f"{data.translate(y_col, cls)}: {recall(y_test, y_test_pred, cls)}")
        print("Training F1:")
        for cls in sorted(set(y_train)):
            print(f"{data.translate(y_col, cls)}: {f1_score(y_train, y_train_pred, cls)}")
        print("Testing F1:")
        for cls in sorted(set(y_train)):
            print(f"{data.translate(y_col, cls)}: {f1_score(y_test, y_test_pred, cls)}")
        print("Training Confusion Matrix:")
        print(confusion_matrix(y_train, y_train_pred, len(set(y_train))))
        print("Testing Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred, len(set(y_train))))
        print(f"Training used {time.time()-start} seconds")
        # clf.print_tree(data, x_cols, y_col) # print tree
        print("==========================")
    return clf


if __name__ == "__main__":
    features = [["Age", "Sex"], ["Age", "Sex", "BP"], ["Age", "Sex", "BP", "Cholesterol"], ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
    y_col = "Drug"
    data = Dataset("../Data/drug200.csv", random_state=42)
    for x_cols in features:
        ent_clf = evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy", verbose=True)
        gini_clf = evaluate(data, x_cols, y_col, max_depth=10, criterion="gini", verbose=True)
        # print(ent_clf)
        # print(gini_clf)