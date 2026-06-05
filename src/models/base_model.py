"""Base Model Template - All models inherit from this"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
import time

class BaseModel(ABC):
    """Abstract base class for all fraud detection models"""
    
    def __init__(self, name, model_dir="models"):
        self.name = name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.training_time = None
        self.metrics = {}
        
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model - must be implemented by child class"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions - must be implemented by child class"""
        pass
    
    @abstractmethod
    def get_feature_importance(self, feature_names):
        """Get feature importance - must be implemented by child class"""
        pass
    
    @abstractmethod
    def save_model(self, filepath):
        """Save model to disk - must be implemented by child class"""
        pass
    
    @abstractmethod
    def load_model(self, filepath):
        """Load model from disk - must be implemented by child class"""
        pass
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
        
        y_pred_class = (y_pred > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        
        self.metrics = {
            'auc': roc_auc_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred_class),
            'precision': tp/(tp+fp) if (tp+fp) > 0 else 0,
            'recall': tp/(tp+fn) if (tp+fn) > 0 else 0,
            'f1': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
        return self.metrics
    
    def print_metrics(self):
        """Print formatted metrics"""
        print(f"\n{'='*50}")
        print(f"📊 {self.name} - PERFORMANCE")
        print(f"{'='*50}")
        print(f"\n📈 AUC Score: {self.metrics['auc']:.4f}")
        print(f"📊 Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"   True Negatives: {self.metrics['tn']:>10,}")
        print(f"   False Positives: {self.metrics['fp']:>10,}")
        print(f"   False Negatives: {self.metrics['fn']:>10,}")
        print(f"   True Positives: {self.metrics['tp']:>10,}")
        print(f"\nPrecision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1 Score: {self.metrics['f1']:.4f}")
        print(f"\n⏱️  Training Time: {self.training_time:.2f} seconds")