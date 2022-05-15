import sys
sys.path.append('..')
import numpy as np
import random
from collections.abc import Callable
from typing import Union
from ..utils.model import BaseModel, BaseClassifier
from ..utils.function import polynomial_kernel, rbf_kernel, sigmoid_kernel


class SVM(BaseModel, BaseClassifier):
    def __init__(self, C:int|float=1.0, 
            kernel:str|Callable[[np.ndarray, np.ndarray], float]='linear', 
            degree:int=3, 
            gamma:str|float='scale', 
            coef0:int|float=0.0, 
            tol:int|float=1e-3,
            heuristic:bool=False,
            max_iter:int=1000):
        super().__init__()
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
        self.heuristic = heuristic
        self.b = 0
        self.E_cache = {}
        self.E_max = -1
        self.E_min = -1

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
            val = polynomial_kernel(self.X_train[i], X_test[j], self.gamma, self.degree, self.coef0)
        elif self.kernel == 'rbf':
            val = rbf_kernel(self.X_train[i], X_test[j], self.gamma)
        elif self.kernel == 'sigmoid':
            val = sigmoid_kernel(self.X_train[i], X_test[j], self.gamma, self.coef0)
        else:
            val = X_test[j, i] # precomputed kernel
        if cached: # cache kernel value
            self.kernel_cache[i, j] = val
            self.cache[i, j] = 1
        return val


    def __random_j(self, i):
        j = random.randint(0, self.X_train.shape[0]-2) # [0, 1, ..., n-2]
        if j >= i:
            j += 1 # [0, 1,..., i-1, i+1,..., n-1]
        return j

    def __error(self, i, cached=False):
        if not cached or i not in self.E_cache:
            return self.__eval(i) - self.y_train[i]
        return self.E_cache[i]

    def __KKT_condition(self, i, cached=False):
        y_i = self.y_train[i]
        E_i = self.__error(i, cached)
        if (E_i*y_i < -self.tol and self.alphas[i] < self.C) or \
                (E_i*y_i > self.tol and self.alphas[i] > 0): # KKT condition
            return True, y_i, E_i
        return False, None, None

    def __optimize(self, i, y_i, E_i, func_j, cached=False):
        j = func_j(i)
        if i == j:
            return 0
        y_j = self.y_train[j]
        E_j = self.__error(j, cached)
        # compute lower and upper bounds
        if y_i != y_j:
            L, H = max(0, self.alphas[j]-self.alphas[i]), \
                    min(self.C, self.C-self.alphas[i]+self.alphas[j])
        else:
            L, H = max(0, self.alphas[i]+self.alphas[j]-self.C), \
                    min(self.C, self.alphas[i]+self.alphas[j])
        if L == H:
            return 0
        k_ii = self.__kernel(i, i) # kernel value
        k_ij = self.__kernel(i, j)
        k_jj = self.__kernel(j, j)
        eta = k_ii + k_jj - 2*k_ij
        if eta <= 0:
            return 0
        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]
        # clip alpha_j
        alpha_j_new = alpha_j_old + y_j*(E_i-E_j)/eta
        alpha_j_new = max(L, alpha_j_new)
        alpha_j_new = min(H, alpha_j_new)
        if abs(alpha_j_new-alpha_j_old) < self.tol:
            return 0
        self.alphas[i] += y_i*y_j*(alpha_j_old-alpha_j_new)
        self.alphas[j] = alpha_j_new
        b_old = self.b
        b_i = self.b - E_i - y_i*(self.alphas[i]-alpha_i_old)*k_ii - y_j*(self.alphas[j]-alpha_j_old)*k_ij
        b_j = self.b - E_j - y_i*(self.alphas[i]-alpha_i_old)*k_ij - y_j*(self.alphas[j]-alpha_j_old)*k_jj
        if 0 < self.alphas[i] < self.C:
            self.b = b_i
        elif 0 < self.alphas[j] < self.C:
            self.b = b_j
        else:
            self.b = (b_i+b_j)/2
        if cached:
            if alpha_i_old == 0 or alpha_i_old == self.C:
                if 0 < self.alphas[i] < self.C:
                    self.E_cache[i] = 0
            elif self.alphas[i] == 0 or self.alphas[i] == self.C:
                self.E_cache.pop(i)
            if alpha_j_old == 0 or alpha_j_old == self.C:
                if 0 < self.alphas[j] < self.C:
                    self.E_cache[j] = 0
            elif self.alphas[j] == 0 or self.alphas[j] == self.C:
                self.E_cache.pop(j)
            for k in self.E_cache:
                self.E_cache[k] = self.E_cache[k] \
                    + y_i*(self.alphas[i]-alpha_i_old)*self.__kernel(i, k)\
                    + y_j*(self.alphas[j]-alpha_j_old)*self.__kernel(j, k) + b_old - self.b
                if self.E_max is None:
                    self.E_max = self.E_cache[k]
                elif self.E_cache[k] > self.E_max:
                    self.E_max = k
                if self.E_min is None:
                    self.E_min = self.E_cache[k]
                elif self.E_cache[k] < self.E_min:
                    self.E_min = k
        return 1

    def __SMO(self):
        n = self.X_train.shape[0]
        self.alphas = np.zeros(n)
        iter = 0
        while iter < self.max_iter:
            alpha_changed = 0 # number of alpha changed
            for i in range(n):
                cond, y_i, E_i = self.__KKT_condition(i)
                if cond:
                    alpha_changed += self.__optimize(i, y_i, E_i, self.__random_j)
            if alpha_changed == 0:
                iter += 1
            else:
                iter = 0
    
    def __heuristic_j(self, i):
        if self.__error(i) >= 0:
            return self.E_min
        return self.E_max

    def __heuristic_optimize(self, i):
        n = self.X_train.shape[0]
        cond, y_i, E_i = self.__KKT_condition(i, True)
        if cond:
            if len(self.E_cache) > 1:
                if self.__optimize(i, y_i, E_i, self.__heuristic_j, True):
                    return 1
            for j in list(self.E_cache.keys()):
                if self.__optimize(i, y_i, E_i, lambda x: j, True):
                    return 1
            for j in range(n):
                if self.__optimize(i, y_i, E_i, lambda x: j, True):
                    return 1
        return 0

    def __SMO_heuristic(self):
        n = self.X_train.shape[0]
        self.alphas = np.zeros(n)
        iter = 0
        check_all = True
        while iter < self.max_iter:
            alpha_changed = 0
            if check_all:
                for i in range(n):
                    alpha_changed += self.__heuristic_optimize(i)
            else:
                for i in list(self.E_cache.keys()):
                    alpha_changed += self.__heuristic_optimize(i)
            if check_all:
                check_all = False
            elif alpha_changed == 0:
                check_all = True
            if alpha_changed == 0:
                iter += 1
            else:
                iter = 0
            
    def fit(self, X, y):
        super().fit(X, y)
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
        if self.heuristic:
            self.__SMO_heuristic()
        else:
            self.__SMO()
        return self
    
    def predict(self, X):
        super().predict(X)
        n, m = X.shape
        pred = np.zeros(n)
        for i in range(n):
            pred[i] = self.__eval(i, X)
        return np.sign(pred)