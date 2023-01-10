import numpy as np


class MyLinearRegression:
    def __init__(self):
        self.coef_ = None 
        self.intercept_ = None
        
    def fit(self, X: 'matrix', y: 'array') -> 'np.array()':
        '''
        ~ fit LinearRegression from sklearn.
        input: 
            X: matrix of size (n, f), where n — nrows in data, f — number of factors
            y: array of size (n, ), where n — nrows in data
        '''
        X = np.array(X)
        y = np.array(y)
        ones = np.ones(X.shape[0])
        X = np.hstack((ones.reshape(-1, 1), X))
        self.coef_ = np.dot((np.linalg.inv(np.dot(X.T, X))), np.dot(X.T, y))
        self.intercept_ = self.coef_[0]
        
        self.coef_ = self.coef_[1:]
        
    def predict(self, X):
        y_pred = np.array(X @ self.coef_ + self.intercept_)
        return y_pred
