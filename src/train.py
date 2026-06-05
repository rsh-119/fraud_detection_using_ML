"""Master Training Orchestrator"""
import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import from models subfolder
from models.base_model import BaseModel
from models.lightgbm_model import LightGBMModel

# ========== SETUP ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

print("="*60)
print("FRAUD DETECTION - MASTER TRAINER")
print("="*60)

# ========== LOAD DATA ==========
print("\n📂 Loading preprocessed data...")
X = pd.read_parquet(DATA_PROCESSED / 'X_train.parquet')
y = pd.read_parquet(DATA_PROCESSED / 'y_train.parquet').squeeze()

print(f"✓ Data: {X.shape[0]:,} rows, {X.shape[1]} features")
print(f"✓ Fraud rate: {y.mean():.4f}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}")

# Calculate class weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"✓ Class weight: {scale_pos_weight:.2f}")

# ========== TRAIN LIGHTGBM ==========
print("\n🚀 Training LightGBM...")

model = LightGBMModel()
model.train(X_train, y_train, X_val, y_val, scale_pos_weight)

# Predict and evaluate
y_pred = model.predict(X_val)
metrics = model.evaluate(y_val, y_pred)
model.print_metrics()

# Save model
model.save_model()

# Feature importance
importance_df = model.get_feature_importance(X.columns)
importance_df.to_csv(MODELS_DIR / "lightgbm_feature_importance.csv", index=False)
print(f"✓ Feature importance saved: lightgbm_feature_importance.csv")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)