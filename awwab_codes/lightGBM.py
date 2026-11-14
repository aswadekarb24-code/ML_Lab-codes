import numpy as np

class SimpleLightGBM:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_bins=16):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_bins = max_bins
        self.trees = []
        self.bin_edges = []
        self.base_pred = 0.0
        self.loss_history = []
        
    def _bin_data(self, X):
        """Histogram binning: converts continuous features into discrete bin indices"""
        n_samples, n_features = X.shape
        binned_X = np.zeros_like(X, dtype=int)
        self.bin_edges = []
        
        for feat in range(n_features):
            col = X[:, feat]
            # Compute quantile-based bin edges
            edges = np.percentile(col, np.linspace(0, 100, self.max_bins + 1))
            edges[0] -= 1e-5 # Expand bounds slightly
            edges[-1] += 1e-5
            self.bin_edges.append(edges)
            # Digitized bin assignments
            binned_X[:, feat] = np.digitize(col, edges) - 1
            binned_X[:, feat] = np.clip(binned_X[:, feat], 0, self.max_bins - 1)
            
        return binned_X

    def fit(self, X, y):
        self.base_pred = np.mean(y)
        y_pred = np.full_like(y, self.base_pred, dtype=float)
        binned_X = self._bin_data(X)
        self.loss_history = []
        
        for i in range(self.n_estimators):
            residuals = y - y_pred
            self.loss_history.append(np.mean(residuals ** 2))
            
            # Find best split using histogram bins (LightGBM core trick)
            best_feat, best_bin, best_mse = None, None, float('inf')
            n_samples, n_features = binned_X.shape
            
            for feat in range(n_features):
                for b in range(self.max_bins):
                    left_mask = binned_X[:, feat] <= b
                    right_mask = ~left_mask
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue
                    mse = np.var(residuals[left_mask]) * np.sum(left_mask) + np.var(residuals[right_mask]) * np.sum(right_mask)
                    if mse < best_mse:
                        best_mse, best_feat, best_bin = mse, feat, b
                        
            if best_feat is None:
                break
                
            left_mask = binned_X[:, best_feat] <= best_bin
            left_val = np.mean(residuals[left_mask]) if np.any(left_mask) else 0
            right_val = np.mean(residuals[~left_mask]) if np.any(~left_mask) else 0
            
            self.trees.append((best_feat, best_bin, left_val, right_val))
            
            update = np.zeros_like(y)
            update[left_mask] = left_val
            update[~left_mask] = right_val
            y_pred += self.lr * update
            
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.full(n_samples, self.base_pred, dtype=float)
        
        # Apply same binning edges used during training
        binned_X = np.zeros_like(X, dtype=int)
        for feat in range(X.shape[1]):
            binned_X[:, feat] = np.digitize(X[:, feat], self.bin_edges[feat]) - 1
            binned_X[:, feat] = np.clip(binned_X[:, feat], 0, self.max_bins - 1)
            
        for feat, b, left_val, right_val in self.trees:
            update = np.where(binned_X[:, feat] <= b, left_val, right_val)
            y_pred += self.lr * update
        return y_pred

from template import run_experiment

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate toy regression data
X_toy, y_toy = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)

# 2. Instantiate model
lgbm_model = SimpleLightGBM(n_estimators=50, learning_rate=0.1)

# 3. Run through your Master Template pipeline
metrics = {
    "MSE": mean_squared_error,
    "R2 Score": r2_score
}

trained_model, preds = run_experiment(
    X=X_toy, 
    y=y_toy, 
    preprocessors=[], 
    model=lgbm_model, 
    metrics_dict=metrics,
    task_type="regression"
)