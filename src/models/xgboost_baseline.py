"""XGBoost Model Training for IEEE-CIS Fraud Detection"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
from pathlib import Path
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# ========== SETUP PATH ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

print("="*60)
print("XGBOOST MODEL TRAINING")
print("="*60)
print(f"\n📁 Project root: {PROJECT_ROOT}")

# ========== 1. LOAD PROCESSED DATA ==========
print("\n1. Loading preprocessed data...")

X_train = pd.read_parquet(DATA_PROCESSED / 'X_train.parquet')
y_train = pd.read_parquet(DATA_PROCESSED / 'y_train.parquet')

if isinstance(y_train, pd.DataFrame):
    y_train = y_train.squeeze()

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ y_train shape: {y_train.shape}")
print(f"✓ Fraud rate: {y_train.mean():.4f}")

# Check for test data
test_file = DATA_PROCESSED / 'X_test.parquet'
has_test = test_file.exists()
if has_test:
    X_test = pd.read_parquet(test_file)
    test_ids = pd.read_parquet(DATA_PROCESSED / 'test_ids.parquet').squeeze()
    print(f"✓ X_test shape: {X_test.shape}")

# ========== 2. SPLIT TRAIN/VALIDATION ==========
print("\n2. Splitting data for validation...")
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"✓ Training set: {X_train_split.shape[0]:,} samples")
print(f"✓ Validation set: {X_val.shape[0]:,} samples")

# ========== 3. TRAIN XGBOOST MODEL ==========
print("\n3. Training XGBoost model...")
start_time = time.time()

# Calculate class weight
neg_count = len(y_train_split[y_train_split == 0])
pos_count = len(y_train_split[y_train_split == 1])
scale_pos_weight = neg_count / pos_count
print(f"✓ Scale pos weight: {scale_pos_weight:.2f}")

# Create DMatrix for faster training
dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
dval = xgb.DMatrix(X_val, label=y_val)

# XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbosity': 0,
    'tree_method': 'hist',  # Faster training
}

# Train with early stopping using watchlist
watchlist = [(dtrain, 'train'), (dval, 'eval')]

model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=False
)

training_time = time.time() - start_time
print(f"✓ Training completed in {training_time:.2f} seconds")
print(f"✓ Best iteration: {model_xgb.best_iteration}")
print(f"✓ Best AUC: {model_xgb.best_score:.4f}")

# ========== 4. EVALUATE ==========
print("\n4. Evaluating model...")

# Predict probabilities
y_val_pred_proba = model_xgb.predict(dval)
y_val_pred_class = (y_val_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_val, y_val_pred_proba)
accuracy = accuracy_score(y_val, y_val_pred_class)
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred_class).ravel()

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"\n📈 AUC Score: {auc:.4f}")
print(f"📊 Accuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:")
print(f"   True Negatives: {tn:>10,}")
print(f"   False Positives: {fp:>10,}")
print(f"   False Negatives: {fn:>10,}")
print(f"   True Positives: {tp:>10,}")
print(f"\nPrecision: {tp/(tp+fp):.4f}")
print(f"Recall: {tp/(tp+fn):.4f}")
print(f"F1 Score: {2*tp/(2*tp+fp+fn):.4f}")

# ========== 5. FEATURE IMPORTANCE ==========
print("\n5. Top 15 Features:")

# Get feature importance
importance_scores = model_xgb.get_score(importance_type='weight')
if not importance_scores:  # If empty, try 'gain'
    importance_scores = model_xgb.get_score(importance_type='gain')

# Create dataframe
importance_df = pd.DataFrame({
    'feature': list(importance_scores.keys()),
    'importance': list(importance_scores.values())
}).sort_values('importance', ascending=False)

# If no importance scores (all features might not be used), use f-score
if len(importance_df) == 0:
    print("   Using feature names from data...")
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': range(len(X_train.columns), 0, -1)
    })

for i in range(min(15, len(importance_df))):
    row = importance_df.iloc[i]
    print(f"   {i+1:2d}. {row['feature'][:40]:40s} : {row['importance']:.4f}")

# ========== 6. SAVE MODEL ==========
print("\n6. Saving model...")
MODELS_DIR.mkdir(exist_ok=True)

# Save in XGBoost native format
model_path = MODELS_DIR / "xgboost_model.json"
model_xgb.save_model(str(model_path))
print(f"✓ Model saved: {model_path}")

# Also save feature importance
importance_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
print(f"✓ Feature importance saved")

# Save metrics
with open(MODELS_DIR / "model_metrics.txt", "w") as f:
    f.write(f"AUC Score: {auc:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {tp/(tp+fp):.4f}\n")
    f.write(f"Recall: {tp/(tp+fn):.4f}\n")
    f.write(f"F1 Score: {2*tp/(2*tp+fp+fn):.4f}\n")
    f.write(f"Best Iteration: {model_xgb.best_iteration}\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")

# ========== 7. TEST PREDICTIONS ==========
if has_test:
    print("\n7. Making test predictions...")
    dtest = xgb.DMatrix(X_test)
    test_predictions = model_xgb.predict(dtest)
    
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': test_predictions
    })
    
    submission_path = SUBMISSIONS_DIR / "submission_basic.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✓ Submission saved: {submission_path}")
    print(f"✓ Prediction range: {test_predictions.min():.4f} - {test_predictions.max():.4f}")
    print(f"✓ Mean prediction: {test_predictions.mean():.4f}")
    
    # Show sample
    print("\n📊 Sample predictions (first 10):")
    print(submission.head(10))

# ========== FINAL SUMMARY ==========
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\n📊 Results Summary:")
print(f"  - Validation AUC: {auc:.4f}")
print(f"  - Features Used: {X_train.shape[1]}")
print(f"  - Training Time: {training_time:.2f} seconds")
print(f"\n✅ Model saved in: {MODELS_DIR}")