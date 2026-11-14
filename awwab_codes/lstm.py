import numpy as np

class SimpleLSTM:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.loss_history = []
        
        # Xavier initialization scaling
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        
        # Weights for 4 gates/components: Forget (f), Input (i), Candidate (c), Output (o)
        # We pack them or define them separately for clarity. Let's define them separately:
        gates = ['f', 'i', 'c', 'o']
        self.W = {}
        self.U = {}
        self.b = {}
        
        for g in gates:
            self.W[g] = np.random.randn(input_dim, hidden_dim) * scale
            self.U[g] = np.random.randn(hidden_dim, hidden_dim) * scale
            self.b[g] = np.zeros((1, hidden_dim))
            # Give input bias a slight positive bump for the forget gate (best practice)
            if g == 'f': 
                self.b[g] = np.ones((1, hidden_dim)) * 1.0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t, h_prev, c_prev):
        """Processes a single time step for LSTM"""
        # 1. Forget Gate (f_t): what to throw away from cell state
        f_t = self._sigmoid(np.dot(x_t, self.W['f']) + np.dot(h_prev, self.U['f']) + self.b['f'])
        
        # 2. Input Gate (i_t): what new information to store in cell state
        i_t = self._sigmoid(np.dot(x_t, self.W['i']) + np.dot(h_prev, self.U['i']) + self.b['i'])
        
        # 3. Candidate Cell State (c_tilde): candidate values to add
        c_tilde = np.tanh(np.dot(x_t, self.W['c']) + np.dot(h_prev, self.U['c']) + self.b['c'])
        
        # 4. Updated Cell State (c_t): conveyor belt update
        c_t = f_t * c_prev + i_t * c_tilde
        
        # 5. Output Gate (o_t): what parts of cell state to output as hidden state
        o_t = self._sigmoid(np.dot(x_t, self.W['o']) + np.dot(h_prev, self.U['o']) + self.b['o'])
        
        # 6. Final Hidden State (h_t)
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t

    def fit(self, X, y, epochs=10):
        # X shape: (samples, time_steps, input_dim)
        n_samples, time_steps, _ = X.shape
        self.loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_samples):
                h = np.zeros((1, self.hidden_dim))
                c = np.zeros((1, self.hidden_dim))
                
                # Forward pass across time steps
                for t in range(time_steps):
                    x_t = X[i, t:t+1, :]
                    h, c = self.forward_step(x_t, h, c)
                
                # Proxy regression loss against target
                pred = h
                target = y[i:i+1] if y.ndim > 1 else np.array([[y[i]] * self.hidden_dim])
                loss = np.mean((pred - target) ** 2)
                epoch_loss += loss
                
                # Simplified gradient descent update step (proxy for backprop)
                grad = (pred - target) * (1.0 / self.hidden_dim)
                self.W['c'] -= self.lr * np.dot(X[i, -1].reshape(-1, 1), grad)
            
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            
        return self

    def predict(self, X):
        n_samples, time_steps, _ = X.shape
        preds = []
        for i in range(n_samples):
            h = np.zeros((1, self.hidden_dim))
            c = np.zeros((1, self.hidden_dim))
            for t in range(time_steps):
                x_t = X[i, t:t+1, :]
                h, c = self.forward_step(x_t, h, c)
            preds.append(h.mean())
        return np.array(preds)

from template import run_experiment
# 1. Generate 3D toy data: (n_samples, time_steps, input_dim)
n_samples, time_steps, input_dim = 100, 5, 4
X_toy = np.random.randn(n_samples, time_steps, input_dim)
y_toy = np.random.randn(n_samples)

# 2. Instantiate LSTM Model
lstm_model = SimpleLSTM(input_dim=input_dim, hidden_dim=8, learning_rate=0.01)

# 3. Run through your Master Template pipeline
from sklearn.metrics import mean_squared_error, r2_score
metrics = {"MSE": mean_squared_error, "R2": r2_score}

trained_model, preds = run_experiment(
    X=X_toy, 
    y=y_toy, 
    preprocessors=[], 
    model=lstm_model, 
    metrics_dict=metrics,
    task_type="regression"
)