import numpy as np

# ==========================================
# 1. SIMPLE CONV2D (CNN)
# ==========================================
class SimpleConv2D:
    def __init__(self, kernel_size=3, out_channels=1, learning_rate=0.01):
        self.k_size = kernel_size
        self.out_channels = out_channels
        self.lr = learning_rate
        self.loss_history = []
        self.kernel = np.random.randn(kernel_size, kernel_size) * 0.1 # Single filter for simplicity
        
    def _relu(self, x):
        return np.maximum(0, x)

    def fit(self, X, y, epochs=5):
        # X shape: (samples, height, width)
        n_samples, h, w = X.shape
        self.loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_samples):
                img = X[i]
                # Convolution operation (sliding dot product)
                out_h = h - self.k_size + 1
                out_w = w - self.k_size + 1
                feature_map = np.zeros((out_h, out_w))
                
                for r in range(out_h):
                    for c in range(out_w):
                        patch = img[r:r+self.k_size, c:c+self.k_size]
                        feature_map[r, c] = np.sum(patch * self.kernel)
                        
                activated = self._relu(feature_map)
                pred = np.mean(activated) # Scalar projection for proxy regression loss
                
                target = y[i]
                loss = (pred - target) ** 2
                epoch_loss += loss
                
                # Simplified gradient descent update proxy
                grad = 2 * (pred - target) * (1.0 / (out_h * out_w))
                self.kernel -= self.lr * grad * np.ones_like(self.kernel)
                
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            
        return self
        
    def predict(self, X):
        n_samples, h, w = X.shape
        preds = []
        out_h = h - self.k_size + 1
        out_w = w - self.k_size + 1
        
        for i in range(n_samples):
            img = X[i]
            feature_map = np.zeros((out_h, out_w))
            for r in range(out_h):
                for c in range(out_w):
                    patch = img[r:r+self.k_size, c:c+self.k_size]
                    feature_map[r, c] = np.sum(patch * self.kernel)
            preds.append(np.mean(self._relu(feature_map)))
        return np.array(preds)


# ==========================================
# 2. SIMPLE VANILLA RNN
# ==========================================
class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.loss_history = []
        
        # Weights for input and hidden states
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros((1, hidden_dim))

    def fit(self, X, y, epochs=10):
        # X shape: (samples, time_steps, input_dim)
        n_samples, time_steps, _ = X.shape
        self.loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_samples):
                h = np.zeros((1, self.hidden_dim))
                
                # Forward pass across time steps
                for t in range(time_steps):
                    x_t = X[i, t:t+1, :]
                    # Vanilla RNN equation: h_t = tanh(W_xh * x_t + W_hh * h_prev + b)
                    h = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
                
                pred = h
                target = y[i:i+1] if y.ndim > 1 else np.array([[y[i]] * self.hidden_dim])
                loss = np.mean((pred - target) ** 2)
                epoch_loss += loss
                
                # Simplified update step
                grad = (pred - target) * (1.0 / self.hidden_dim)
                self.W_xh -= self.lr * np.dot(X[i, -1].reshape(-1, 1), grad)
                
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            
        return self

    def predict(self, X):
        n_samples, time_steps, _ = X.shape
        preds = []
        for i in range(n_samples):
            h = np.zeros((1, self.hidden_dim))
            for t in range(time_steps):
                x_t = X[i, t:t+1, :]
                h = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            preds.append(h.mean())
        return np.array(preds)

from template import run_experiment
from sklearn.metrics import mean_squared_error, r2_score

# --- Testing CNN ---
print("Running CNN Experiment...")
X_cnn = np.random.randn(50, 10, 10) # 50 images of size 10x10
y_cnn = np.random.randn(50)
cnn_model = SimpleConv2D(kernel_size=3, learning_rate=0.01)

run_experiment(
    X=X_cnn, y=y_cnn, preprocessors=[], 
    model=cnn_model, metrics_dict={"MSE": mean_squared_error}, task_type="regression"
)

# --- Testing Vanilla RNN ---
print("\nRunning Vanilla RNN Experiment...")
X_rnn = np.random.randn(50, 5, 4) # 50 sequences, 5 time steps, 4 features
y_rnn = np.random.randn(50)
rnn_model = SimpleRNN(input_dim=4, hidden_dim=8, learning_rate=0.01)

run_experiment(
    X=X_rnn, y=y_rnn, preprocessors=[], 
    model=rnn_model, metrics_dict={"MSE": mean_squared_error}, task_type="regression"
)