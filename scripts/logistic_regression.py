import numpy as np

import numpy as np

class MyLogisticRegression:
    def __init__(self, learning_rate=0.1, n_iters=1000, lambda_=0.01):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def compute_loss(self, y_true, y_pred):

        epsilon = 1e-9
        loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

        loss += self.lambda_ * np.sum(self.weights ** 2) / 2
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        y_one_hot = self._one_hot(y, n_classes)

        for _ in range(self.n_iters):
            logits = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(logits)

            error = y_pred - y_one_hot
            dw = (np.dot(X.T, error) / n_samples) + self.lambda_ * self.weights
            db = np.mean(error, axis=0, keepdims=True)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)
