import numpy as np
import pandas as pd

class SimpleDecisionTreeNode:
    def __init__(self, feature=None, threshold=None, label=None, isLeaf=False, children=None):
        self.feature = feature          # Feature index or name to split on
        self.threshold = threshold      # Threshold if continuous (None if categorical)
        self.label = label              # Prediction label if leaf
        self.isLeaf = isLeaf
        self.children = children if children is not None else {} # For categorical splits: {val: node}
        self.left = None                # For continuous left split (<= threshold)
        self.right = None               # For continuous right split (> threshold)

class SimpleDecisionTree:
    def __init__(self, choice="infoGain", max_depth=5, sample_limit=2):
        self.root = None
        self.choice = choice
        self.max_depth = max_depth
        self.sample_limit = sample_limit
        self.loss_history = []          # Dummy attribute to satisfy master template checks

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_feat, best_thresh = None, None
        base_metric = self._entropy(y) if self.choice in ["infoGain", "gainRatio"] else self._gini(y)

        for feat in range(n_features):
            x_column = X[:, feat]
            unique_vals = np.unique(x_column)
            
            # If feature is continuous or has many unique values, consider midpoints as thresholds
            if len(unique_vals) > 10:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            else:
                thresholds = unique_vals

            for thresh in thresholds:
                # Check if it's treated as a continuous threshold split
                left_mask = x_column <= thresh
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Weighted child impurity
                n = len(y)
                child_impurity = (np.sum(left_mask)/n) * (self._entropy(y[left_mask]) if self.choice in ["infoGain", "gainRatio"] else self._gini(y[left_mask])) + \
                                 (np.sum(right_mask)/n) * (self._entropy(y[right_mask]) if self.choice in ["infoGain", "gainRatio"] else self._gini(y[right_mask]))
                
                gain = base_metric - child_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        majority_label = np.argmax(np.bincount(y.astype(int)))

        # Stopping criteria
        if len(np.unique(y)) == 1 or depth >= self.max_depth or n_samples < self.sample_limit:
            return SimpleDecisionTreeNode(label=majority_label, isLeaf=True)

        feat, thresh, gain = self._find_best_split(X, y)
        
        if feat is None or gain <= 1e-7:
            return SimpleDecisionTreeNode(label=majority_label, isLeaf=True)

        # Split data continuously
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        node = SimpleDecisionTreeNode(feature=feat, threshold=thresh)
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node

    def fit(self, X, y):
        # Convert pandas DataFrame/Series to numpy arrays cleanly if passed
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_row(self, node, x):
        if node.isLeaf:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        else:
            return self._predict_row(node.right, x)

    def predict(self, X):
        if hasattr(X, "values"):
            X = X.values
        return np.array([self._predict_row(self.root, x) for x in X])

from template import run_experiment
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 1. Generate toy classification data (mix of continuous features)
X_toy, y_toy = make_classification(n_samples=300, n_features=4, random_state=42)

# 2. Instantiate our streamlined Decision Tree
dt_model = SimpleDecisionTree(choice="infoGain", max_depth=4)

# 3. Run through your Master Template pipeline
metrics = {"Accuracy": accuracy_score}

trained_model, preds = run_experiment(
    X=X_toy, 
    y=y_toy, 
    preprocessors=[], # Tree models don't strictly require scaling
    model=dt_model, 
    metrics_dict=metrics,
    task_type="classification"
)