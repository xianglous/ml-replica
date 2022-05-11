import pandas as pd
import numpy as np
from utils.preprocessing import Standardizer, MinMaxScaler


class Dataset:
    def __init__(self, filename, ratio=0.8, random_state=None):
        self.data = pd.read_csv(filename)
        self.num_cols = self.data._get_numeric_data().columns
        self.cate_cols = self.data.columns.difference(self.num_cols)
        self.cate_map = {}
        for col in self.cate_cols:
            self.data[col], self.cate_map[col] = pd.factorize(self.data[col])
        if isinstance(ratio, float):
            if ratio > 1 or ratio < 0:
                raise ValueError("ratio should be between 0 and 1")
            self.train_ratio = ratio
            self.val_ratio = 0.0
        elif isinstance(ratio, tuple) or isinstance(ratio, list):
            if len(ratio) < 2 or len(ratio) > 3:
                raise ValueError("ratio should be a tuple or list with 2 or 3 elements")
            total = sum(ratio)
            ratio = [rat / total for rat in ratio]
            self.train_ratio = ratio[0]
            if len(ratio) == 2:
                self.val_ratio = 0.0
            else:
                self.val_ratio = ratio[1]
        else:
            raise ValueError("ratio should be a float or a tuple or a list")
        train_data = self.data.sample(frac=self.train_ratio, random_state=random_state)
        val_test_data = self.data.drop(train_data.index)
        val_data = val_test_data.sample(frac=self.val_ratio, random_state=random_state)
        test_data = val_test_data.drop(val_data.index)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def transform(self, cols, method="standardize"):
        if isinstance(cols, str):
            cols = [cols]
        cols = [col for col in cols if col in self.num_cols]
        if callable(method):
            self.transformer = method
        else:
            if method == "standardize":
                self.transformer = Standardizer(self.train_data[cols].to_numpy())
            elif method == "min_max_scale":
                self.transformer = MinMaxScaler(self.train_data[cols].to_numpy())
            else:
                raise ValueError("method should be one of 'standardize', 'min_max_scale' or a callable")
        self.train_data[cols] = self.transformer(self.train_data[cols].to_numpy())
        self.val_data[cols] = self.transformer(self.val_data[cols].to_numpy())
        self.test_data[cols] = self.transformer(self.test_data[cols].to_numpy())
    
    def get_split(self, x_cols, y_col, val=False):
        X_train, y_train = self.get_train(x_cols, y_col)
        X_test, y_test = self.get_test(x_cols, y_col)
        if val:
            X_val, y_val = self.get_val(x_cols, y_col)
            return X_train, y_train, X_val, y_val, X_test, y_test
        return X_train, y_train, X_test, y_test

    def get_train(self, x_cols, y_col):
        return self.train_data[x_cols].to_numpy(), self.train_data[y_col].to_numpy()
    
    def get_val(self, x_cols, y_col):
        return self.val_data[x_cols].to_numpy(), self.val_data[y_col].to_numpy()
    
    def get_test(self, x_cols, y_col):
        return self.test_data[x_cols].to_numpy(), self.test_data[y_col].to_numpy()

    def translate(self, col, vals):
        if col not in self.cate_cols:
            raise ValueError("column should be one of the categorical columns")
        return self.cate_map[col][vals]

    def predict(self, model, df, x_cols):
        if isinstance(x_cols, str):
            x_cols = [x_cols]
        n_cols = [col for col in x_cols if col in self.num_cols]
        c_cols = [col for col in x_cols if col in self.cate_cols]
        df[n_cols] = self.transformer(df[n_cols].to_numpy())
        for col in c_cols:
            df[col] = self.translate(col, df[col].to_numpy())
        return model(df[x_cols].to_numpy())
        

# def prepare_data(filename, x_cols, y_col, X_transform="standard", y_transform=None, ratio=0.8, random_state=42):
#     df = pd.read_csv(filename)
#     train_df = df.sample(frac=ratio, random_state=random_state)
#     test_df = df.drop(train_df.index)
#     X_train, y_train = train_df[x_cols].to_numpy(), train_df[y_col].to_numpy()
#     X_test, y_test = test_df[x_cols].to_numpy(), test_df[y_col].to_numpy()
#     if X_transform == "standard":
#         X_trans = standardizer(X_train)
#     elif X_transform == "min_max":
#         X_trans = min_max_scaler(X_train)
#     else:
#         X_trans = None
    
#     if y_transform == "standard":
#         y_trans = standardizer(None, y_train)
#     elif y_transform == "perceptron":
#         y_trans = perceptronizer
#     elif y_transform == "min_max":
#         y_trans = min_max_scaler(None, y_train)
#     else:
#         y_trans = None
    
#     if X_trans is not None:
#         X_train = X_trans(X_train)
#         X_test = X_trans(X_test)
#     if y_trans is not None:
#         y_train = y_trans(y_train)
#         y_test = y_trans(y_test)

#     return X_train, y_train, X_test, y_test


def model_str(x_cols, y_col, offset_enabled=False, method="linear"):
    if method == "linear":
        return f"Model: {y_col} = {'w_0 + ' if offset_enabled else ''}" +\
            f"{' + '.join([f'w_{i+1} * {x_cols[i]}' for i in range(len(x_cols))])}"
    elif method == "logistic":
        return f"Model: {y_col} = 1 / (1 + exp(-{'w_0 + ' if offset_enabled else ''}" +\
            f"{' + '.join([f'w_{i+1} * {x_cols[i]}' for i in range(len(x_cols))])}))"
