import numpy as np

class SimpleGRU:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.loss_history = []
        
        # Xavier/Normal Initialization for Weights (Input to Gates)
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_z = np.random.randn(input_dim, hidden_dim) * scale
        self.W_r = np.random.randn(input_dim, hidden_dim) * scale
        self.W_h = np.random.randn(input_dim, hidden_dim) * scale
        
        # Recurrent Weights (Hidden State to Gates)
        self.U_z = np.random.randn(hidden_dim, hidden_dim) * scale
        self.U_r = np.random.randn(hidden_dim, hidden_dim) * scale
        self.U_h = np.random.randn(hidden_dim, hidden_dim) * scale
        
        # Biases
        self.b_z = np.zeros((1, hidden_dim))
        self.b_r = np.zeros((1, hidden_dim))
        self.b_h = np.zeros((1, hidden_dim))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t, h_prev):
        """Processes a single time step of the sequence"""
        # 1. Update Gate z_t: decides how much past state to keep
        z_t = self._sigmoid(np.dot(x_t, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z)
        
        # 2. Reset Gate r_t: decides how much past state to forget
        r_t = self._sigmoid(np.dot(x_t, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r)
        
        # 3. Candidate Hidden State h_tilde: proposed content using reset gate
        h_tilde = np.tanh(np.dot(x_t, self.W_h) + np.dot(r_t * h_prev, self.U_h) + self.b_h)
        
        # 4. Final Hidden State h_t: linear interpolation between old state and candidate
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t, z_t, r_t, h_tilde

    def fit(self, X, y, epochs=10):
        # X shape: (samples, time_steps, input_dim)
        # y shape: (samples, hidden_dim) or target mapping
        n_samples, time_steps, _ = X.shape
        self.loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_samples):
                h = np.zeros((1, self.hidden_dim)) # Initialize hidden state to 0
                
                # Forward pass across time steps for sequence i
                for t in range(time_steps):
                    x_t = X[i, t:t+1, :]
                    h, _, _, _ = self.forward_step(x_t, h)
                
                # Dummy regression loss against target y[i] (MSE-like proxy)
                pred = h
                target = y[i:i+1] if y.ndim > 1 else np.array([[y[i]] * self.hidden_dim])
                loss = np.mean((pred - target) ** 2)
                epoch_loss += loss
                
                # Simplified gradient descent update step (approximate backprop proxy for brevity)
                grad = (pred - target) * (1.0 / self.hidden_dim)
                self.W_h -= self.lr * np.dot(X[i, -1].reshape(-1, 1), grad)
            
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
                h, _, _, _ = self.forward_step(x_t, h)
            preds.append(h.mean()) # Return scalar projection per sequence
        return np.array(preds)
    
from template import run_experiment

# 1. Generate 3D toy data for RNN/GRU: (n_samples, time_steps, features)
n_samples, time_steps, input_dim = 100, 5, 4
X_toy = np.random.randn(n_samples, time_steps, input_dim)
y_toy = np.random.randn(n_samples) # Continuous targets

# 2. Instantiate GRU Model
gru_model = SimpleGRU(input_dim=input_dim, hidden_dim=8, learning_rate=0.01)

# 3. Run through your Master Template pipeline
from sklearn.metrics import mean_squared_error, r2_score
metrics = {"MSE": mean_squared_error, "R2": r2_score}

trained_model, preds = run_experiment(
    X=X_toy, 
    y=y_toy, 
    preprocessors=[], # Already 3D, skip standard 2D scalers for raw test
    model=gru_model, 
    metrics_dict=metrics,
    task_type="regression"
)