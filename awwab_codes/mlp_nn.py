import numpy as np

class SimpleMLPClassifier:
    def __init__(self, hidden_dim=16, learning_rate=0.1, epochs=100, random_state=None):
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.loss_history = []
        self.loss_ = [] # Alias to support template check for sklearn-like models too
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def _relu(self, x):
        return np.maximum(0, x)
        
    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Handle binary vs multi-class output adjustment for simplicity (using binary output here)
        out_dim = 1 if n_classes <= 2 else n_classes
        
        # Xavier/He Initialization
        scale_in = np.sqrt(2.0 / n_features)
        scale_hid = np.sqrt(2.0 / self.hidden_dim)
        
        self.W1 = np.random.randn(n_features, self.hidden_dim) * scale_in
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, out_dim) * scale_hid
        self.b2 = np.zeros((1, out_dim))
        
        self.loss_history = []
        
        for epoch in range(self.epochs):
            # --- FORWARD PASS ---
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            preds = self._sigmoid(z2) # Output probabilities
            
            # --- LOSS COMPUTATION (Binary Cross-Entropy) ---
            y_reshaped = y.reshape(-1, 1)
            # Clip predictions to prevent log(0)
            preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
            loss = -np.mean(y_reshaped * np.log(preds_clipped) + (1 - y_reshaped) * np.log(1 - preds_clipped))
            
            self.loss_history.append(loss)
            
            # --- BACKWARD PASS (Backpropagation) ---
            error = preds - y_reshaped
            
            # Gradients for Layer 2
            dW2 = np.dot(a1.T, error) / n_samples
            db2 = np.sum(error, axis=0, keepdims=True) / n_samples
            
            # Gradients for Layer 1
            error_hidden = np.dot(error, self.W2.T) * self._relu_deriv(a1)
            dW1 = np.dot(X.T, error_hidden) / n_samples
            db1 = np.sum(error_hidden, axis=0, keepdims=True) / n_samples
            
            # --- GRADIENT DESCENT UPDATE ---
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            
        # Mirror loss history to sklearn-style attribute just in case
        self.loss_ = self.loss_history
        return self

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = self._sigmoid(z2)
        return (probs >= 0.5).astype(int).flatten()
    
from template import run_experiment

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Generate toy classification data
X_toy, y_toy = make_classification(n_samples=300, n_features=6, random_state=42)

# 2. Instantiate our Custom MLP Classifier
mlp_model = SimpleMLPClassifier(hidden_dim=12, learning_rate=0.1, epochs=150, random_state=42)

# 3. Run through your Master Template pipeline
metrics = {
    "Accuracy": accuracy_score
}

trained_model, preds = run_experiment(
    X=X_toy, 
    y=y_toy, 
    preprocessors=[StandardScaler()], # Neural networks require scaled inputs!
    model=mlp_model, 
    metrics_dict=metrics,
    task_type="classification"
)
