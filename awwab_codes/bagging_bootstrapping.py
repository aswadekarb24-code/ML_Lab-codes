import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

class CustomBaggingClassifier:
    def __init__(self, n_estimators=10, max_samples=1.0, random_state=None):
        self.n_estimators = n_estimators       # Number of bootstrap trees
        self.max_samples = max_samples         # Fraction of samples to draw per tree
        self.random_state = random_state
        self.models = []                       # Store individual trained trees
        self.loss_history = []                 # Empty dummy loss to satisfy template checks
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples = X.shape[0]
        sample_size = int(self.max_samples * n_samples)
        self.models = []
        
        for i in range(self.n_estimators):
            # 1. BOOTSTRAPPING: Sample indices with replacement
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            
            # Handle pandas DataFrames vs numpy arrays cleanly
            if hasattr(X, "iloc"):
                X_sample = X.iloc[indices]
                y_sample = y.iloc[indices]
            else:
                X_sample = X[indices]
                y_sample = y[indices]
                
            # 2. Train a base learner (Deep Decision Tree usually)
            base_tree = DecisionTreeClassifier(random_state=self.random_state + i if self.random_state else None)
            base_tree.fit(X_sample, y_sample)
            
            self.models.append(base_tree)
            
        return self
        
    def predict(self, X):
        # Collect predictions from all bootstrap models
        predictions = np.array([tree.predict(X) for tree in self.models]) # Shape: (n_estimators, n_test_samples)
        
        # AGGREGATION: Majority vote across trees for each sample
        # mode returns the most frequent value along the columns (axis=0)
        maj_vote, _ = mode(predictions, axis=0)
        return maj_vote.flatten()

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from template import run_experiment

# 1. Generate toy binary classification data
X_toy, y_toy = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)

# 2. Instantiate our Custom Bagging Classifier
bagging_model = CustomBaggingClassifier(n_estimators=20, max_samples=0.8, random_state=42)

# 3. Run through your Master Template pipeline
metrics = {
    "Accuracy": accuracy_score
}

trained_model, preds = run_experiment(
    X=X_toy, 
    y=y_toy, 
    preprocessors=[StandardScaler()], 
    model=bagging_model, 
    metrics_dict=metrics,
    task_type="classification"
)