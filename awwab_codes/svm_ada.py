import numpy as np

# ==========================================
# 1. SIMPLE LINEAR SVM (Gradient Descent)
# ==========================================
class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=200):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # Regularization
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.loss_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 1. Safely convert {0, 1} labels to {-1, 1} for SVM math
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for idx, x_i in enumerate(X):
                # Hinge loss margin condition: y_i * (w * x_i - b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                # Compute hinge loss for tracking
                loss = max(0, 1 - y_[idx] * (np.dot(x_i, self.w) - self.b))
                epoch_loss += loss
                
                # Gradient updates
                if condition:
                    grad_w = self.lambda_param * self.w
                    grad_b = 0
                else:
                    grad_w = self.lambda_param * self.w - np.dot(y_[idx], x_i)
                    grad_b = -y_[idx]
                    
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
                
            self.loss_history.append(epoch_loss / n_samples)
            
        return self

    def predict(self, X):
        # 2. Compute decision boundary and map back to {0, 1}
        linear_output = np.dot(X, self.w) - self.b
        preds = np.sign(linear_output)
        # Map {-1, 0, 1} back to binary {0, 1} (treating 0 as class 1 or 0 safely)
        return np.where(preds < 0, 0, 1)

class KernelSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear', gamma=0.1, degree=3, coef0=1.0):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        # Kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        # alpha = weights per sample
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []

    def _get_kernel(self, X1, X2):
        """Calculates the Kernel matrix between X1 and X2."""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            # Efficient vectorized RBF calculation: exp(-gamma * ||X1 - X2||^2)
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1)
            dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * dist_sq)
        else:
            raise ValueError("Invalid kernel. Choose 'linear', 'poly', or 'rbf'.")

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.X_train = X
        self.y_train = np.where(y <= 0, -1, 1)

        # Initialize alphas
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Precompute the Kernel (Gram) matrix for training
        K = self._get_kernel(X, X)

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                # The prediction for X[idx] is sum(alpha * y * K(X, X_train)) + b
                margin = np.sum(self.alpha * self.y_train * K[:, idx]) + self.b
                condition = self.y_train[idx] * margin >= 1

                if condition:
                    # Regularization only
                    self.alpha -= self.lr * (2 * self.lambda_param * self.alpha)
                else:
                    # Update alpha for the misclassified sample and adjust bias
                    self.alpha[idx] += self.lr
                    self.alpha -= self.lr * (2 * self.lambda_param * self.alpha)
                    self.b += self.lr * self.y_train[idx]

            Y_pred = self.predict_raw(X, K_precomputed=K)
            self.loss_history.append(self.hinge_loss(Y_pred, self.y_train))
        self.support_ = np.where(np.abs(self.alpha) > 1e-5)[0]
        self.support_vectors_ = X[self.support_]

    def hinge_loss(self, Y_pred_raw, Y_true):
        # Standard SVM loss function
        distances = 1 - Y_true * Y_pred_raw
        distances[distances < 0] = 0
        hinge = np.mean(distances)
        return hinge

    def predict_raw(self, X, K_precomputed=None):
        if K_precomputed is not None:
            K = K_precomputed
        else:
            K = self._get_kernel(X, self.X_train)

        # Prediction formula: sum(alpha_i * y_i * K(x_i, x_test)) + b
        return np.dot(K, self.alpha * self.y_train) + self.b

    def predict(self, X):
        return np.where(np.sign(self.predict_raw(X)) < 0, 0, 1)

# ==========================================
# 2. SIMPLE ADABOOST CLASSIFIER
# ==========================================
class SimpleAdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.alphas = []
        self.stumps = []
        self.loss_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize uniform sample weights
        w = np.ones(n_samples) / n_samples
        self.loss_history = []
        
        for i in range(self.n_estimators):
            stump = {}
            min_error = float('inf')
            best_feat, best_thresh, best_polarity = None, None, 1
            
            # Find the best decision stump
            for feat in range(n_features):
                thresholds = np.unique(X[:, feat])
                for thresh in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X[:, feat] < thresh] = -1
                        else:
                            predictions[X[:, feat] > thresh] = -1
                            
                        # Weighted error calculation
                        error = np.sum(w[predictions != y_])
                        
                        if error < min_error:
                            min_error = error
                            best_feat = feat
                            best_thresh = thresh
                            best_polarity = polarity
                            
            # Avoid division by zero or negative weights
            min_error = max(min_error, 1e-10)
            if min_error > 0.5: # Stop if worse than random guessing
                break
                
            # Compute alpha (stump weight)
            alpha = 0.5 * np.log((1.0 - min_error) / min_error)
            
            # Get best predictions for updating sample weights
            predictions = np.ones(n_samples)
            if best_polarity == 1:
                predictions[X[:, best_feat] < best_thresh] = -1
            else:
                predictions[X[:, best_feat] > best_thresh] = -1
                
            # Update sample weights (exponentially penalize misclassifications)
            w *= np.exp(-alpha * y_ * predictions)
            w /= np.sum(w) # Re-normalize weights
            
            self.alphas.append(alpha)
            self.stumps.append((best_feat, best_thresh, best_polarity))
            self.loss_history.append(min_error)
            
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        stump_preds = np.zeros(n_samples)
        
        for alpha, (feat, thresh, polarity) in zip(self.alphas, self.stumps):
            predictions = np.ones(n_samples)
            if polarity == 1:
                predictions[X[:, feat] < thresh] = -1
            else:
                predictions[X[:, feat] > thresh] = -1
            stump_preds += alpha * predictions
            
        final_preds = np.sign(stump_preds)
        return np.where(final_preds <= 0, 0, 1)

from template import run_experiment
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate toy data
X_toy, y_toy = make_classification(n_samples=300, n_features=4, random_state=42)

# Instantiate patched SVM with a fine-tuned small learning rate
svm_model = SimpleSVM(learning_rate=0.05, lambda_param=0.01, epochs=3000)

kernelsvm = KernelSVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000, kernel='rbf', gamma=0.5)
run_experiment(
    X=X_toy, y=y_toy, preprocessors=[StandardScaler()], 
    model=svm_model, metrics_dict={"Accuracy": accuracy_score}, task_type="classification"
)
run_experiment(
    X=X_toy, y=y_toy, preprocessors=[StandardScaler()], 
    model=kernelsvm, metrics_dict={"Accuracy": accuracy_score}, task_type="classification"
)
# Generate toy binary classification dataset
X_toy, y_toy = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)

# --- Test AdaBoost ---
print("\nRunning AdaBoost Experiment...")
ada_model = SimpleAdaBoost(n_estimators=10)
run_experiment(
    X=X_toy, y=y_toy, preprocessors=[StandardScaler()], 
    model=ada_model, metrics_dict={"Accuracy": accuracy_score}, task_type="classification"
)