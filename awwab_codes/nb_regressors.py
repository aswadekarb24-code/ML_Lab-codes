import numpy as np

# ==========================================
# 1. GAUSSIAN NAIVE BAYES
# ==========================================
class SimpleGaussianNB:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
        self.loss_history = [] # Dummy list to satisfy template plot check
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            # Calculate mean and variance per feature for each class
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9  # Add small smoothing term to avoid division by zero
            self.priors[c] = X_c.shape[0] / X.shape[0] # Class prior probability P(C)
        return self
        
    def _pdf(self, class_idx, x):
        """Gaussian Probability Density Function"""
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                # Use log probabilities to prevent numerical underflow
                prior = np.log(self.priors[c])
                conditional = np.sum(np.log(self._pdf(c, x)))
                posterior = prior + conditional
                posteriors.append(posterior)
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)


# ==========================================
# 2. LINEAR REGRESSION (Gradient Descent)
# ==========================================
class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.loss_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for epoch in range(self.epochs):
            # Forward pass: y_pred = Xw + b
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute Mean Squared Error (MSE) Loss
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # Gradients calculation
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Gradient descent update step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# ==========================================
# 3. LOGISTIC REGRESSION
# ==========================================
class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.loss_history = []
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for epoch in range(self.epochs):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply Sigmoid activation function
            y_pred = self._sigmoid(linear_model)
            
            # Binary Cross-Entropy Loss
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.loss_history.append(loss)
            
            # Gradients calculation
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Gradient descent update step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        return self

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        # Threshold at 0.5 for binary classification
        return (y_pred >= 0.5).astype(int)

from template import run_experiment
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# --- Test 1: Gaussian Naive Bayes ---
print("--- Naive Bayes Experiment ---")
X_class, y_class = make_classification(n_samples=200, n_features=4, random_state=42)
nb_model = SimpleGaussianNB()
run_experiment(
    X=X_class, y=y_class, preprocessors=[], 
    model=nb_model, metrics_dict={"Accuracy": accuracy_score}, task_type="classification"
)

# --- Test 2: Linear Regression ---
print("\n--- Linear Regression Experiment ---")
X_reg, y_reg = make_regression(n_samples=200, n_features=3, noise=0.1, random_state=42)
lin_model = SimpleLinearRegression(learning_rate=0.05, epochs=100)
run_experiment(
    X=X_reg, y=y_reg, preprocessors=[StandardScaler()], 
    model=lin_model, metrics_dict={"MSE": mean_squared_error, "R2": r2_score}, task_type="regression"
)

# --- Test 3: Logistic Regression ---
print("\n--- Logistic Regression Experiment ---")
log_model = SimpleLogisticRegression(learning_rate=0.1, epochs=150)
run_experiment(
    X=X_class, y=y_class, preprocessors=[StandardScaler()], 
    model=log_model, metrics_dict={"Accuracy": accuracy_score}, task_type="classification"
)