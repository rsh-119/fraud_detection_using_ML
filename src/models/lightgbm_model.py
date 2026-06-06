"""LightGBM Model Training - Standalone Script"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import lightgbm as lgb
from pathlib import Path
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# ========== SETUP PATH ==========
# Go up 3 levels: models -> src -> project_root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

print("="*60)
print("LIGHTGBM MODEL TRAINING")
print("="*60)
print(f"\n📁 Project root: {PROJECT_ROOT}")
print(f"📁 Data path: {DATA_PROCESSED}")

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

# ========== 3. TRAIN LIGHTGBM MODEL ==========
print("\n3. Training LightGBM model...")
start_time = time.time()

# Calculate class weight
scale_pos_weight = (y_train_split == 0).sum() / (y_train_split == 1).sum()
print(f"✓ Scale pos weight: {scale_pos_weight:.2f}")

# LightGBM parameters (optimized for fraud detection)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,  # Use all CPU cores
}

# Create model
model_lgb = lgb.LGBMClassifier(**params)

# Train with early stopping
model_lgb.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

training_time = time.time() - start_time
print(f"✓ Training completed in {training_time:.2f} seconds")
print(f"✓ Best iteration: {model_lgb.best_iteration_}")

# ========== 4. EVALUATE ==========
print("\n4. Evaluating model...")

# Predict probabilities
y_val_pred_proba = model_lgb.predict_proba(X_val)[:, 1]
y_val_pred_class = (y_val_pred_proba > 0.5).astype(int)

# Calculate metrics
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

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_lgb.feature_importances_
}).sort_values('importance', ascending=False)

for i in range(min(15, len(importance_df))):
    row = importance_df.iloc[i]
    print(f"   {i+1:2d}. {row['feature'][:40]:40s} : {row['importance']:.0f}")

# ========== 6. SAVE MODEL ==========
print("\n6. Saving model...")
MODELS_DIR.mkdir(exist_ok=True)

# Save model
model_path = MODELS_DIR / "lightgbm_model.pkl"
joblib.dump(model_lgb, model_path)
print(f"✓ Model saved: {model_path}")

# Save feature importance
importance_df.to_csv(MODELS_DIR / "lightgbm_feature_importance.csv", index=False)
print(f"✓ Feature importance saved: lightgbm_feature_importance.csv")

# Save metrics
with open(MODELS_DIR / "lightgbm_model_metrics.txt", "w") as f:
    f.write(f"AUC Score: {auc:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {tp/(tp+fp):.4f}\n")
    f.write(f"Recall: {tp/(tp+fn):.4f}\n")
    f.write(f"F1 Score: {2*tp/(2*tp+fp+fn):.4f}\n")
    f.write(f"Best Iteration: {model_lgb.best_iteration_}\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
print(f"✓ Metrics saved: lightgbm_model_metrics.txt")

# ========== 7. TEST PREDICTIONS ==========
if has_test:
    print("\n7. Making test predictions...")
    test_predictions = model_lgb.predict_proba(X_test)[:, 1]
    
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': test_predictions
    })
    
    submission_path = SUBMISSIONS_DIR / "submission_lightgbm.csv"
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

# Compare with XGBoost if available
print("\n" + "="*60)
print("📊 COMPARISON WITH XGBOOST")
print("="*60)

xgb_metrics_file = MODELS_DIR / "model_metrics.txt"
if xgb_metrics_file.exists():
    with open(xgb_metrics_file, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if 'AUC Score' in line:
                xgb_auc = line.split(':')[1].strip()
                print(f"   XGBoost AUC: {xgb_auc}")
                print(f"   LightGBM AUC: {auc:.4f}")
                if auc > float(xgb_auc):
                    print(f"   🏆 LightGBM is better by {auc - float(xgb_auc):.4f}")
                else:
                    print(f"   🏆 XGBoost is better by {float(xgb_auc) - auc:.4f}")
                break
else:
    print("   XGBoost metrics not found (run xgboost_baseline.py first)")