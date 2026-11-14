import numpy as np

class SimpleGradientBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.base_pred = 0.0
        self.loss_history = []
        
    def _mse_split(self, X, y):
        # Find best split for a simple regression stump (heart of the tree)
        best_val, best_feat, best_mse = float('inf'), None, float('inf')
        n_samples, n_features = X.shape
        
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                mse = np.var(y[left_mask]) * sum(left_mask) + np.var(y[right_mask]) * sum(right_mask)
                if mse < best_mse:
                    best_mse, best_feat, best_val = mse, feat, thresh
                    
        return best_feat, best_val

    def fit(self, X, y):
        self.base_pred = np.mean(y)
        y_pred = np.full_like(y, self.base_pred, dtype=float)
        self.loss_history = []
        
        for i in range(self.n_estimators):
            # 1. Compute gradients (residuals for MSE loss)
            residuals = y - y_pred
            self.loss_history.append(np.mean(residuals ** 2))
            
            # 2. Fit a simple stump to residuals
            feat, thresh = self._mse_split(X, residuals)
            if feat is None:
                break
                
            left_mask = X[:, feat] <= thresh
            left_val = np.mean(residuals[left_mask]) if np.any(left_mask) else 0
            right_val = np.mean(residuals[~left_mask]) if np.any(~left_mask) else 0
            
            self.trees.append((feat, thresh, left_val, right_val))
            
            # 3. Update predictions with learning rate shrinkage
            update = np.zeros_like(y)
            update[left_mask] = left_val
            update[~left_mask] = right_val
            y_pred += self.learning_rate * update
            
        return self
        
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_pred, dtype=float)
        for feat, thresh, left_val, right_val in self.trees:
            update = np.where(X[:, feat] <= thresh, left_val, right_val)
            y_pred += self.learning_rate * update
        return y_pred
    
from template import run_experiment

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate toy regression data
X_toy, y_toy = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)

# 2. Instantiate model
xgb_model = SimpleGradientBoost(n_estimators=50, learning_rate=0.1)

# 3. Run through your Master Template pipeline
metrics = {
    "MSE": mean_squared_error,
    "R2 Score": r2_score
}

trained_model, preds = run_experiment(
    X=X_toy, 
    y=y_toy, 
    preprocessors=[], 
    model=xgb_model, 
    metrics_dict=metrics,
    task_type="regression"
)