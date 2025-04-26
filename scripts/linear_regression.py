import numpy as np

class LinearRegressionClosedForm:
    def __init__(self):
        self.weights = None 
    
    def fit(self, X, y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        self.weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    
    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return np.round(X_bias @ self.weights)