"""XGBoost Model Training for IEEE-CIS Fraud Detection - Tuned Version"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve
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
print("XGBOOST MODEL TRAINING (TUNED)")
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

# ── FIX: compute from actual split counts, never hardcode ──
neg_count = (y_train_split == 0).sum()
pos_count = (y_train_split == 1).sum()
scale_pos_weight = neg_count / pos_count          # was hardcoded to 40
print(f"✓ Neg samples:      {neg_count:,}")
print(f"✓ Pos samples:      {pos_count:,}")
print(f"✓ Scale pos weight: {scale_pos_weight:.2f}  (computed from data)")

# Create DMatrix for faster training
dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
dval   = xgb.DMatrix(X_val,         label=y_val)

# XGBoost parameters
params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'max_depth':        6,
    'learning_rate':    0.05,       # was 0.1 — lower lr + more rounds = better
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,         # added — prevents overfitting on rare fraud
    'reg_alpha':        0.1,        # added — L1 regularization
    'reg_lambda':       1.0,        # added — L2 regularization
    'scale_pos_weight': scale_pos_weight,
    'random_state':     42,
    'verbosity':        0,
    'tree_method':      'hist',
}

watchlist = [(dtrain, 'train'), (dval, 'eval')]

model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,           # was 500 — gives early stopping real room
    evals=watchlist,
    early_stopping_rounds=100,      # was 50 — more patience at lower lr
    verbose_eval=100,               # print every 100 rounds so you can watch progress
)

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds")
print(f"✓ Best iteration:   {model_xgb.best_iteration}")
print(f"✓ Best AUC:         {model_xgb.best_score:.4f}")

# ========== 4. EVALUATE ==========
print("\n4. Evaluating model...")

y_val_pred_proba = model_xgb.predict(dval)

auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"\n📈 AUC Score: {auc:.4f}")

# ========== 5. THRESHOLD TUNING ==========
print("\n5. Tuning decision threshold for best F1...")

precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred_proba)

f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx       = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1        = f1_scores[best_idx]

print(f"✓ Best threshold: {best_threshold:.4f}")
print(f"✓ Best F1:        {best_f1:.4f}")

# Also report threshold that gives ~80% recall
target_recall = 0.80
recall_idx = np.where(recalls >= target_recall)[0]
if len(recall_idx) > 0:
    best_recall_idx   = recall_idx[np.argmax(precisions[recall_idx])]
    recall_threshold  = thresholds[min(best_recall_idx, len(thresholds) - 1)]
    print(f"✓ Threshold for ~80% recall: {recall_threshold:.4f} "
          f"(Precision: {precisions[best_recall_idx]:.4f}, "
          f"Recall: {recalls[best_recall_idx]:.4f})")

# Evaluate at best-F1 threshold
y_val_pred_class = (y_val_pred_proba > best_threshold).astype(int)
accuracy = accuracy_score(y_val, y_val_pred_class)
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred_class).ravel()

print("\n" + "="*60)
print("MODEL PERFORMANCE (at best-F1 threshold)")
print("="*60)
print(f"\n📈 AUC Score:  {auc:.4f}")
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

# ========== 6. FEATURE IMPORTANCE ==========
print("\n6. Top 15 Features:")

importance_scores = model_xgb.get_score(importance_type='gain')   # gain > weight
if not importance_scores:
    importance_scores = model_xgb.get_score(importance_type='weight')

importance_df = pd.DataFrame({
    'feature':    list(importance_scores.keys()),
    'importance': list(importance_scores.values())
}).sort_values('importance', ascending=False)

if len(importance_df) == 0:
    importance_df = pd.DataFrame({
        'feature':    X_train.columns,
        'importance': range(len(X_train.columns), 0, -1)
    })

for i in range(min(15, len(importance_df))):
    row = importance_df.iloc[i]
    print(f"   {i+1:2d}. {row['feature'][:40]:40s} : {row['importance']:.4f}")

# ========== 7. SAVE MODEL ==========
print("\n7. Saving model...")
MODELS_DIR.mkdir(exist_ok=True)

model_path = MODELS_DIR / "xgboost_model.json"
model_xgb.save_model(str(model_path))
print(f"✓ Model saved: {model_path}")

importance_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
print(f"✓ Feature importance saved")

# Save best threshold so ensemble can load it
np.save(MODELS_DIR / "xgboost_threshold.npy", best_threshold)
print(f"✓ Best threshold saved: xgboost_threshold.npy")

with open(MODELS_DIR / "model_metrics.txt", "w") as f:
    f.write(f"AUC Score: {auc:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {tp/(tp+fp):.4f}\n")
    f.write(f"Recall: {tp/(tp+fn):.4f}\n")
    f.write(f"F1 Score: {2*tp/(2*tp+fp+fn):.4f}\n")
    f.write(f"Best Threshold: {best_threshold:.4f}\n")
    f.write(f"Best Iteration: {model_xgb.best_iteration}\n")
    f.write(f"Scale Pos Weight: {scale_pos_weight:.2f}\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
print(f"✓ Metrics saved: model_metrics.txt")

# ========== 8. TEST PREDICTIONS ==========
if has_test:
    print("\n8. Making test predictions...")
    dtest = xgb.DMatrix(X_test)
    test_predictions = model_xgb.predict(dtest)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud':       test_predictions
    })

    submission_path = SUBMISSIONS_DIR / "submission_basic.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✓ Submission saved: {submission_path}")
    print(f"✓ Prediction range: {test_predictions.min():.4f} - {test_predictions.max():.4f}")
    print(f"✓ Mean prediction:  {test_predictions.mean():.4f}")

    print("\n📊 Sample predictions (first 10):")
    print(submission.head(10))

# ========== FINAL SUMMARY ==========
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\n📊 Results Summary:")
print(f"  - Validation AUC:   {auc:.4f}")
print(f"  - Best Threshold:   {best_threshold:.4f}")
print(f"  - Features Used:    {X_train.shape[1]}")
print(f"  - Best Iteration:   {model_xgb.best_iteration}")
print(f"  - Scale Pos Weight: {scale_pos_weight:.2f}  (computed)")
print(f"  - Training Time:    {training_time:.2f} seconds")
print(f"\n✅ Model saved in: {MODELS_DIR}")