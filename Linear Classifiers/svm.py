from subprocess import call
import numpy as np
import random
from collections.abc import Callable
from typing import Union
from utils import prepare_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SVM:
    def __init__(self, C:Union[int, float]=1.0, 
                kernel:Union[str, Callable]='linear', 
                degree:int=3, 
                gamma:Union[str, float]='scale', 
                coef0:Union[int, float]=0.0, 
                tol:Union[int, float]=1e-3,
                max_iter:int=1000):

        if (not callable(kernel)) and \
                (kernel not in {"linear", "poly", "rbf", "sigmoid", "precomputed"}):
            raise ValueError("Invalid kernel")
        if degree <= 0:
            raise ValueError("Invalid degree")
        if C <= 0:
            raise ValueError("Invalid C")
        if isinstance(gamma, str):
            if gamma != 'scale' and gamma != 'auto':
                raise ValueError("Invalid gamma")
        elif gamma <= 0:
            raise ValueError("Invalid gamma")
        if coef0 < 0:
            raise ValueError("Invalid coef0")

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.b = 0

    def __eval(self, i, X=None):
        if X is None:
            X = self.X_train # use training data
        val = self.b
        for j in range(self.X_train.shape[0]):
            val += self.alphas[j]*self.y_train[j]*self.__kernel(j, i, X)
        return val
    
    def __kernel(self, i, j, X_test=None):
        cached = False
        if X_test is None:
            X_test = self.X_train
            cached = True # cache while training
        if cached and self.cache[i, j]:
            return self.kernel_cache[i, j] # use cached value
        if callable(self.kernel): # user-defined kernel
            val = self.kernel(self.X_train[i], X_test[j])
        elif self.kernel == 'linear':
            val = self.X_train[i].T@X_test[j]
        elif self.kernel == 'poly':
            val = (self.gamma*self.X_train[i].T@X_test[j]+self.coef0)**self.degree
        elif self.kernel == 'rbf':
            val = np.exp(-self.gamma*(np.sum((self.X_train[i]-X_test[j])**2)))
        elif self.kernel == 'sigmoid':
            val = np.tanh(self.gamma*self.X_train[i].T@X_test[j]+self.coef0)
        else:
            val = X_test[j, i] # precomputed kernel
        if cached: # cache kernel value
            self.kernel_cache[i, j] = val
            self.cache[i, j] = 1
        return val


    def __select_j(self, i):
        j = random.randint(0, self.X_train.shape[0]-2) # [0, 1, ..., n-2]
        if j >= i:
            j += 1 # [0, 1,..., i-1, i+1,..., n-1]
        return j

    def __SMO(self):
        n = self.X_train.shape[0]
        self.alphas = np.zeros(n)
        iter = 0
        while iter < self.max_iter:
            alpha_changed = 0 # number of alpha changed
            for i in range(n):
                if self.alphas[i] != 0:
                    continue
                j = self.__select_j(i) # select random j
                pred_i, pred_j = self.__eval(i), self.__eval(j)
                E_i, E_j = pred_i - self.y_train[i], pred_j - self.y_train[j] # error
                k_ii = self.__kernel(i, i) # kernel value
                k_ij = self.__kernel(i, j)
                k_jj = self.__kernel(j, j)
                # compute lower and upper bounds
                if self.y_train[i] != self.y_train[j]:
                    L, H = max(0, self.alphas[j]-self.alphas[i]), min(self.C, self.C-self.alphas[i]+self.alphas[j])
                else:
                    L, H = max(0, self.alphas[i]+self.alphas[j]-self.C), min(self.C, self.alphas[i]+self.alphas[j])
                if L == H:
                    continue
                eta = k_ii + k_jj - 2*k_ij #
                if eta <= 0:
                    continue
                alpha_i_old = self.alphas[i]
                alpha_j_old = self.alphas[j]
                # clip alpha_j
                alpha_j_new = alpha_j_old + self.y_train[j]*(E_i-E_j)/eta
                alpha_j_new = max(L, alpha_j_new)
                alpha_j_new = min(H, alpha_j_new)
                if abs(alpha_j_new-alpha_j_old) < self.tol:
                    continue
                self.alphas[i] += self.y_train[i]*self.y_train[j]*(alpha_j_old-alpha_j_new)
                self.alphas[j] = alpha_j_new
                b_i = self.b - E_i - self.y_train[i]*(self.alphas[i]-alpha_i_old)*k_ii - self.y_train[j]*(self.alphas[j]-alpha_j_old)*k_ij
                b_j = self.b - E_j - self.y_train[i]*(self.alphas[i]-alpha_i_old)*k_ij - self.y_train[j]*(self.alphas[j]-alpha_j_old)*k_jj
                if 0 < self.alphas[i] < self.C:
                    self.b = b_i
                elif 0 < self.alphas[j] < self.C:
                    self.b = b_j
                else:
                    self.b = (b_i+b_j)/2
                alpha_changed += 1
            if alpha_changed == 0:
                iter += 1
            else:
                iter = 0
        
            
    def fit(self, X, y):
        n, m = X.shape
        if self.kernel == 'precomputed' and n != m:
            raise ValueError("Gram matrix must be a square matrix")
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self.gamma = 1.0 / (n*X.var())
            else:
                self.gamma = 1.0 / n
        self.X_train = X
        self.y_train = y
        self.kernel_cache = np.zeros((n, n))
        self.cache = np.zeros((n, n))
        if self.kernel == 'precomputed':
            self.kernel_cache = X
            self.cache = np.ones((n, n))
        self.__SMO()
    
    def predict(self, X):
        n, m = X.shape
        pred = np.zeros(n)
        for i in range(n):
            pred[i] = self.__eval(i, X)
        return np.sign(pred)
    

def evaluate(filename, x_cols, y_col, kernel="linear", tol=1e-3, max_iter=1000):
    X_train, y_train, X_test, y_test = prepare_data(filename, x_cols, y_col)
    print("==========================")
    print("Kernel:", kernel)
    clf = SVM(kernel=kernel, tol=tol, max_iter=max_iter)
    if kernel == 'precomputed':
        X_test = X_test@X_train.T
        X_train = X_train@X_train.T
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("==========================")


if __name__ == "__main__":
    x_cols = ["age", "interest"]
    y_col = "success"
    for kernel in ['linear', 'poly', 'rbf', 'precomputed']:
         evaluate("Data/binary_classification.csv", x_cols, y_col, kernel)
    
