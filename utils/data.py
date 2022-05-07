import pandas as pd


def prepare_data(filename, x_cols, y_col, ratio=0.8, random_state=42):
    df = pd.read_csv(filename)
    df[y_col] = df[y_col] * 2 - 1 # -1 or 1
    train_df = df.sample(frac=ratio, random_state=random_state)
    test_df = df.drop(train_df.index)
    X_train, y_train = train_df[x_cols].to_numpy(), train_df[y_col].to_numpy()
    X_test, y_test = test_df[x_cols].to_numpy(), test_df[y_col].to_numpy()
    return X_train, y_train, X_test, y_test


def model_str(x_cols, y_col, offset_enabled=False):
    return f"Model: {y_col} = {'w_0 + ' if offset_enabled else ''}" +\
           f"{' + '.join([f'w_{i+1} * {x_cols[i]}' for i in range(len(x_cols))])}"
