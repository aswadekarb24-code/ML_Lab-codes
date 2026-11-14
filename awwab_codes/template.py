# ==========================================
# 0. STANDARD IMPORTS
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score # Add others as needed

# ==========================================
# 1. WRITE YOUR ALGORITHM CLASS HERE (If Custom)
# ==========================================
class CustomModel:
    def __init__(self, **kwargs):
        # Set hyperparameters here
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.loss_history = []
        
    def fit(self, X, y):
        # Implement training logic here
        # Make sure to append epoch/iteration losses to self.loss_history
        pass
        
    def predict(self, X):
        # Implement prediction logic here
        pass

# ==========================================
# 2. UNIVERSAL PIPELINE & EVALUATION FUNCTION
# ==========================================
def run_experiment(X, y, preprocessors, model, metrics_dict, task_type="classification", test_size=0.3, random_state=42):
    # A. Preprocessing Pipeline
    X_processed = X.copy()
    for prep in preprocessors:
        X_processed = prep.fit_transform(X_processed)
        
    # B. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )
    
    # C. Model Training & Prediction
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # D. Metric Evaluation
    print("=== MODEL PERFORMANCE EVALUATION ===")
    for metric_name, metric_func in metrics_dict.items():
        score = metric_func(y_test, y_pred)
        print(f"{metric_name}: {score:.4f}")
        
    # E. Loss Convergence Plot (Cleaned up for Exams)
    if hasattr(model, "loss_history") and len(model.loss_history) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(model.loss_history, color='blue', lw=2)
        plt.xlabel('Iterations / Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Convergence History')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif hasattr(model, "loss_") and len(model.loss_) > 0: # For sklearn models like MLPClassifier
        plt.figure(figsize=(8, 4))
        plt.plot(model.loss_, color='blue', lw=2)
        plt.xlabel('Iterations / Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Convergence History')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    return model, y_pred