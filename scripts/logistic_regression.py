import numpy as np

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = - (1/m) * np.sum(y_true*np.log(y_pred + 1e-15) + (1 - y_true)*np.log(1 - y_pred + 1e-15))
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                linear_output = np.dot(X_batch, self.weights) + self.bias
                y_predicted = self.sigmoid(linear_output)

                error = y_predicted - y_batch
                dw = (1/len(y_batch)) * np.dot(X_batch.T, error)
                db = (1/len(y_batch)) * np.sum(error)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            if epoch % 10 == 0:
                loss = self.compute_loss(y, self.sigmoid(np.dot(X, self.weights) + self.bias))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
