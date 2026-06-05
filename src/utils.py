"""Shared utility functions"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

def load_data(data_path="data/processed"):
    """Load preprocessed data"""
    data_dir = Path(data_path)
    X = pd.read_parquet(data_dir / 'X_train.parquet')
    y = pd.read_parquet(data_dir / 'y_train.parquet').squeeze()
    return X, y

def load_test_data(data_path="data/processed"):
    """Load test data if exists"""
    data_dir = Path(data_path)
    test_file = data_dir / 'X_test.parquet'
    if test_file.exists():
        X_test = pd.read_parquet(test_file)
        test_ids = pd.read_parquet(data_dir / 'test_ids.parquet').squeeze()
        return X_test, test_ids
    return None, None

def save_submission(predictions, test_ids, model_name, submission_dir="submissions"):
    """Save submission file"""
    sub_dir = Path(submission_dir)
    sub_dir.mkdir(exist_ok=True)
    
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': predictions
    })
    submission_path = sub_dir / f"submission_{model_name}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✓ Submission saved: {submission_path}")
    return submission_path