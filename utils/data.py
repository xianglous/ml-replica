import pandas as pd
from utils.preprocessing import standardizer, min_max_scaler, perceptronizer


def prepare_data(filename, x_cols, y_col, X_transform="standard", y_transform=None, ratio=0.8, random_state=42):
    df = pd.read_csv(filename)
    train_df = df.sample(frac=ratio, random_state=random_state)
    test_df = df.drop(train_df.index)
    X_train, y_train = train_df[x_cols].to_numpy(), train_df[y_col].to_numpy()
    X_test, y_test = test_df[x_cols].to_numpy(), test_df[y_col].to_numpy()
    if X_transform == "standard":
        X_trans = standardizer(X_train)
    elif X_transform == "min_max":
        X_trans = min_max_scaler(X_train)
    else:
        X_trans = None
    
    if y_transform == "standard":
        y_trans = standardizer(None, y_train)
    elif y_transform == "perceptron":
        y_trans = perceptronizer
    elif y_transform == "min_max":
        y_trans = min_max_scaler(None, y_train)
    else:
        y_trans = None
    
    if X_trans is not None:
        X_train = X_trans(X_train)
        X_test = X_trans(X_test)
    if y_trans is not None:
        y_train = y_trans(y_train)
        y_test = y_trans(y_test)

    return X_train, y_train, X_test, y_test


def model_str(x_cols, y_col, offset_enabled=False, method="linear"):
    if method == "linear":
        return f"Model: {y_col} = {'w_0 + ' if offset_enabled else ''}" +\
            f"{' + '.join([f'w_{i+1} * {x_cols[i]}' for i in range(len(x_cols))])}"
    elif method == "logistic":
        return f"Model: {y_col} = 1 / (1 + exp(-{'w_0 + ' if offset_enabled else ''}" +\
            f"{' + '.join([f'w_{i+1} * {x_cols[i]}' for i in range(len(x_cols))])}))"
