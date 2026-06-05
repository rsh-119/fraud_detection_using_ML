"""Ensemble Model - Combine XGBoost and LightGBM predictions"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

print("="*60)
print("ENSEMBLE MODEL - XGBoost + LightGBM")
print("="*60)

# ========== 1. LOAD DATA ==========
print("\n1. Loading data...")
X = pd.read_parquet(DATA_PROCESSED / 'X_train.parquet')
y = pd.read_parquet(DATA_PROCESSED / 'y_train.parquet').squeeze()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train):,}, Val: {len(X_val):,}")

# ========== 2. LOAD TRAINED MODELS ==========
print("\n2. Loading trained models...")

# Load XGBoost
xgb_model = xgb.Booster()
xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))
print("✓ XGBoost model loaded")

# Load LightGBM
lgb_model = joblib.load(MODELS_DIR / "lightgbm_model.pkl")
print("✓ LightGBM model loaded")

# ========== 3. MAKE PREDICTIONS ==========
print("\n3. Making predictions...")

# XGBoost predictions
dval = xgb.DMatrix(X_val)
xgb_pred = xgb_model.predict(dval)

# LightGBM predictions
lgb_pred = lgb_model.predict_proba(X_val)[:, 1]

# ========== 4. TRY DIFFERENT ENSEMBLE METHODS ==========
print("\n4. Testing ensemble methods...")

# Method 1: Simple Average
ensemble_avg = (xgb_pred + lgb_pred) / 2
auc_avg = roc_auc_score(y_val, ensemble_avg)

# Method 2: Weighted Average (optimize weights)
best_auc = 0
best_weight = 0.5
for weight in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    ensemble_weighted = (weight * xgb_pred) + ((1-weight) * lgb_pred)
    auc = roc_auc_score(y_val, ensemble_weighted)
    if auc > best_auc:
        best_auc = auc
        best_weight = weight

# Method 3: Max probability (take highest fraud probability)
ensemble_max = np.maximum(xgb_pred, lgb_pred)
auc_max = roc_auc_score(y_val, ensemble_max)

# Method 4: Rank averaging
from scipy.stats import rankdata
rank_xgb = rankdata(xgb_pred) / len(xgb_pred)
rank_lgb = rankdata(lgb_pred) / len(lgb_pred)
ensemble_rank = (rank_xgb + rank_lgb) / 2
auc_rank = roc_auc_score(y_val, ensemble_rank)

# ========== 5. RESULTS ==========
print("\n" + "="*60)
print("📊 ENSEMBLE RESULTS")
print("="*60)
print(f"\nIndividual Models:")
print(f"   XGBoost : 0.9584")
print(f"   LightGBM: 0.9241")
print(f"\nEnsemble Methods:")
print(f"   Simple Average     : {auc_avg:.4f}")
print(f"   Weighted Average   : {best_auc:.4f} (XGB weight: {best_weight:.1f})")
print(f"   Max Probability    : {auc_max:.4f}")
print(f"   Rank Averaging     : {auc_rank:.4f}")

# Best ensemble method
best_auc = max(auc_avg, best_auc, auc_max, auc_rank)
best_method = ""
if best_auc == auc_avg: best_method = "Simple Average"
elif best_auc == best_auc: best_method = f"Weighted Average (XGB:{best_weight:.1f})"
elif best_auc == auc_max: best_method = "Max Probability"
else: best_method = "Rank Averaging"

print(f"\n🏆 BEST ENSEMBLE: {best_method} with AUC: {best_auc:.4f}")
print(f"   Improvement over XGBoost: +{best_auc - 0.9584:.4f}")

# ========== 6. SAVE ENSEMBLE PREDICTIONS ==========
print("\n6. Saving ensemble predictions...")

# Use best ensemble method for final predictions
if best_method == "Simple Average":
    final_ensemble = (xgb_pred + lgb_pred) / 2
elif "Weighted" in best_method:
    final_ensemble = (best_weight * xgb_pred) + ((1-best_weight) * lgb_pred)
elif best_method == "Max Probability":
    final_ensemble = np.maximum(xgb_pred, lgb_pred)
else:
    final_ensemble = ensemble_rank

# Save validation predictions
val_results = pd.DataFrame({
    'XGBoost': xgb_pred,
    'LightGBM': lgb_pred,
    'Ensemble': final_ensemble,
    'Actual': y_val
})
val_results.to_csv(MODELS_DIR / "ensemble_predictions.csv", index=False)
print(f"✓ Saved: {MODELS_DIR / 'ensemble_predictions.csv'}")

# ========== 7. MAKE TEST PREDICTIONS ==========
test_file = DATA_PROCESSED / 'X_test.parquet'
if test_file.exists():
    print("\n7. Making test predictions with ensemble...")
    X_test = pd.read_parquet(test_file)
    test_ids = pd.read_parquet(DATA_PROCESSED / 'test_ids.parquet').squeeze()
    
    # Predict with both models
    dtest = xgb.DMatrix(X_test)
    xgb_test = xgb_model.predict(dtest)
    lgb_test = lgb_model.predict_proba(X_test)[:, 1]
    
    # Apply best ensemble method
    if best_method == "Simple Average":
        ensemble_test = (xgb_test + lgb_test) / 2
    elif "Weighted" in best_method:
        ensemble_test = (best_weight * xgb_test) + ((1-best_weight) * lgb_test)
    elif best_method == "Max Probability":
        ensemble_test = np.maximum(xgb_test, lgb_test)
    else:
        rank_xgb_test = rankdata(xgb_test) / len(xgb_test)
        rank_lgb_test = rankdata(lgb_test) / len(lgb_test)
        ensemble_test = (rank_xgb_test + rank_lgb_test) / 2
    
    # Save submission
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': ensemble_test
    })
    
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission_path = SUBMISSIONS_DIR / "submission_ensemble.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✓ Ensemble submission saved: {submission_path}")
    print(f"✓ Prediction range: {ensemble_test.min():.4f} - {ensemble_test.max():.4f}")

print("\n" + "="*60)
print("ENSEMBLE COMPLETE!")
print("="*60)
print(f"\n✅ Best AUC: {best_auc:.4f}")
print(f"✅ Improvement: +{best_auc - 0.9584:.4f} over XGBoost")