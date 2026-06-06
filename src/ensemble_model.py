"""Ensemble: XGBoost + LightGBM Blender for IEEE-CIS Fraud Detection"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ========== SETUP PATH ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

print("="*60)
print("ENSEMBLE: XGBoost + LightGBM BLENDER")
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

test_file = DATA_PROCESSED / 'X_test.parquet'
has_test = test_file.exists()
if has_test:
    X_test = pd.read_parquet(test_file)
    test_ids = pd.read_parquet(DATA_PROCESSED / 'test_ids.parquet').squeeze()
    print(f"✓ X_test shape: {X_test.shape}")

# ========== 2. SPLIT (same seed as training scripts) ==========
print("\n2. Recreating validation split (same seed as training)...")
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)
print(f"✓ Validation set: {X_val.shape[0]:,} samples")

# ========== 3. LOAD SAVED MODELS ==========
print("\n3. Loading saved models...")

# --- XGBoost ---
xgb_path = MODELS_DIR / "xgboost_model.json"
if not xgb_path.exists():
    raise FileNotFoundError(
        f"XGBoost model not found at {xgb_path}. "
        "Run xgboost_baseline.py first."
    )
model_xgb = xgb.Booster()
model_xgb.load_model(str(xgb_path))
print(f"✓ XGBoost loaded from: {xgb_path}")

# --- LightGBM (prefer tuned, fall back to original) ---
lgb_tuned_path = MODELS_DIR / "lightgbm_model_tuned.pkl"
lgb_original_path = MODELS_DIR / "lightgbm_model.pkl"

if lgb_tuned_path.exists():
    model_lgb = joblib.load(lgb_tuned_path)
    lgb_label = "LightGBM (tuned)"
    print(f"✓ {lgb_label} loaded from: {lgb_tuned_path}")
elif lgb_original_path.exists():
    model_lgb = joblib.load(lgb_original_path)
    lgb_label = "LightGBM (original)"
    print(f"✓ {lgb_label} loaded from: {lgb_original_path}")
else:
    raise FileNotFoundError(
        f"No LightGBM model found. Run lightgbm_model_tuned.py first."
    )

# ========== 4. GET INDIVIDUAL PREDICTIONS ON VALIDATION SET ==========
print("\n4. Generating individual model predictions on validation set...")

# XGBoost predictions
dval = xgb.DMatrix(X_val)
xgb_val_proba = model_xgb.predict(dval)

# LightGBM predictions
lgb_val_proba = model_lgb.predict_proba(X_val)[:, 1]

# Individual AUCs
xgb_auc = roc_auc_score(y_val, xgb_val_proba)
lgb_auc = roc_auc_score(y_val, lgb_val_proba)

print(f"✓ XGBoost  validation AUC: {xgb_auc:.4f}")
print(f"✓ {lgb_label} validation AUC: {lgb_auc:.4f}")

# ========== 5. FIND OPTIMAL ENSEMBLE WEIGHTS ==========
print("\n5. Finding optimal blend weights...")

best_auc = 0
best_w = 0.5

# Grid search over XGBoost weight (LightGBM weight = 1 - w)
for w in np.arange(0.0, 1.01, 0.05):
    blended = w * xgb_val_proba + (1 - w) * lgb_val_proba
    auc = roc_auc_score(y_val, blended)
    if auc > best_auc:
        best_auc = auc
        best_w = w

xgb_weight = best_w
lgb_weight = 1 - best_w

print(f"✓ Best XGBoost weight:  {xgb_weight:.2f}")
print(f"✓ Best LightGBM weight: {lgb_weight:.2f}")
print(f"✓ Blended AUC at best weights: {best_auc:.4f}")

# Final blended probabilities on validation
ensemble_val_proba = xgb_weight * xgb_val_proba + lgb_weight * lgb_val_proba

# ========== 6. THRESHOLD TUNING ON ENSEMBLE ==========
print("\n6. Tuning decision threshold on ensemble probabilities...")

precisions, recalls, thresholds = precision_recall_curve(y_val, ensemble_val_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"✓ Best threshold (max F1): {best_threshold:.4f}")

# Threshold for ~80% recall
target_recall = 0.80
recall_idx = np.where(recalls >= target_recall)[0]
if len(recall_idx) > 0:
    best_recall_idx = recall_idx[np.argmax(precisions[recall_idx])]
    recall_threshold = thresholds[min(best_recall_idx, len(thresholds)-1)]
    print(f"✓ Threshold for ~80% recall: {recall_threshold:.4f} "
          f"(Precision: {precisions[best_recall_idx]:.4f}, Recall: {recalls[best_recall_idx]:.4f})")

# ========== 7. FINAL EVALUATION ==========
print("\n7. Evaluating ensemble...")

y_ensemble_class = (ensemble_val_proba > best_threshold).astype(int)
accuracy = accuracy_score(y_val, y_ensemble_class)
tn, fp, fn, tp = confusion_matrix(y_val, y_ensemble_class).ravel()

print("\n" + "="*60)
print("ENSEMBLE PERFORMANCE")
print("="*60)
print(f"\n📈 AUC Score:  {best_auc:.4f}")
print(f"📊 Accuracy:   {accuracy:.4f}")
print(f"🎯 Threshold:  {best_threshold:.4f}")
print(f"\nConfusion Matrix:")
print(f"   True Negatives:  {tn:>10,}")
print(f"   False Positives: {fp:>10,}")
print(f"   False Negatives: {fn:>10,}")
print(f"   True Positives:  {tp:>10,}")
print(f"\nPrecision: {tp/(tp+fp):.4f}")
print(f"Recall:    {tp/(tp+fn):.4f}")
print(f"F1 Score:  {2*tp/(2*tp+fp+fn):.4f}")

# ========== 8. COMPARE ALL MODELS ==========
print("\n" + "="*60)
print("📊 FULL COMPARISON")
print("="*60)

rows = [
    ("XGBoost (0.5 thresh)",  xgb_auc,  roc_auc_score(y_val, xgb_val_proba)),
    (f"{lgb_label} (0.5 thresh)", lgb_auc, roc_auc_score(y_val, lgb_val_proba)),
    ("Ensemble (tuned thresh)", best_auc, best_auc),
]

print(f"\n{'Model':<35} {'AUC':>8}")
print("-" * 45)
for name, auc_val, _ in rows:
    marker = " 🏆" if auc_val == max(r[1] for r in rows) else ""
    print(f"   {name:<33} {auc_val:.4f}{marker}")

improvement_over_xgb = best_auc - xgb_auc
improvement_over_lgb = best_auc - lgb_auc
print(f"\n   Ensemble vs XGBoost:    {improvement_over_xgb:+.4f}")
print(f"   Ensemble vs {lgb_label}: {improvement_over_lgb:+.4f}")

# ========== 9. SAVE ENSEMBLE CONFIG ==========
print("\n8. Saving ensemble config and metrics...")
MODELS_DIR.mkdir(exist_ok=True)

ensemble_config = {
    'xgb_weight': float(xgb_weight),
    'lgb_weight': float(lgb_weight),
    'best_threshold': float(best_threshold),
    'ensemble_auc': float(best_auc),
    'xgb_auc': float(xgb_auc),
    'lgb_auc': float(lgb_auc),
    'lgb_model_used': lgb_label,
}

import json
with open(MODELS_DIR / "ensemble_config.json", "w") as f:
    json.dump(ensemble_config, f, indent=2)
print(f"✓ Ensemble config saved: ensemble_config.json")

with open(MODELS_DIR / "ensemble_metrics.txt", "w") as f:
    f.write(f"AUC Score: {best_auc:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {tp/(tp+fp):.4f}\n")
    f.write(f"Recall: {tp/(tp+fn):.4f}\n")
    f.write(f"F1 Score: {2*tp/(2*tp+fp+fn):.4f}\n")
    f.write(f"Best Threshold: {best_threshold:.4f}\n")
    f.write(f"XGBoost Weight: {xgb_weight:.2f}\n")
    f.write(f"LightGBM Weight: {lgb_weight:.2f}\n")
print(f"✓ Metrics saved: ensemble_metrics.txt")

# ========== 10. TEST PREDICTIONS ==========
if has_test:
    print("\n9. Making test predictions...")
    start_time = time.time()

    dtest = xgb.DMatrix(X_test)
    xgb_test_proba = model_xgb.predict(dtest)
    lgb_test_proba = model_lgb.predict_proba(X_test)[:, 1]

    ensemble_test_proba = xgb_weight * xgb_test_proba + lgb_weight * lgb_test_proba

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': ensemble_test_proba
    })

    submission_path = SUBMISSIONS_DIR / "submission_ensemble.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✓ Submission saved: {submission_path}")
    print(f"✓ Prediction range: {ensemble_test_proba.min():.4f} - {ensemble_test_proba.max():.4f}")
    print(f"✓ Mean prediction:  {ensemble_test_proba.mean():.4f}")
    print(f"✓ Inference time:   {time.time() - start_time:.2f} seconds")

    print("\n📊 Sample predictions (first 10):")
    print(submission.head(10))

# ========== FINAL SUMMARY ==========
print("\n" + "="*60)
print("ENSEMBLE COMPLETE!")
print("="*60)
print(f"\n📊 Final Results:")
print(f"  - Ensemble AUC:     {best_auc:.4f}")
print(f"  - XGBoost weight:   {xgb_weight:.2f}")
print(f"  - LightGBM weight:  {lgb_weight:.2f}")
print(f"  - Best threshold:   {best_threshold:.4f}")
print(f"  - F1 Score:         {2*tp/(2*tp+fp+fn):.4f}")
print(f"\n✅ Config & metrics saved in: {MODELS_DIR}")
if has_test:
    print(f"✅ Submission saved in:       {SUBMISSIONS_DIR}")