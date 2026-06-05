"""Compare XGBoost vs LightGBM results"""
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

print("="*60)
print("XGBOOST vs LIGHTGBM - COMPARISON")
print("="*60)

results = {}

# Load XGBoost metrics
xgb_file = MODELS_DIR / "model_metrics.txt"
if xgb_file.exists():
    with open(xgb_file, 'r') as f:
        for line in f:
            if 'AUC Score:' in line:
                results['XGBoost'] = float(line.split(':')[1].strip())
                break
    print(f"\n✓ Loaded XGBoost results from: {xgb_file}")
else:
    print(f"\n❌ XGBoost metrics not found at: {xgb_file}")
    print("   Run: python src/models/xgboost_baseline.py first")

# Load LightGBM metrics
lgb_file = MODELS_DIR / "lightgbm_model_metrics.txt"
if lgb_file.exists():
    with open(lgb_file, 'r') as f:
        for line in f:
            if 'AUC Score:' in line:
                results['LightGBM'] = float(line.split(':')[1].strip())
                break
    print(f"✓ Loaded LightGBM results from: {lgb_file}")
else:
    print(f"\n❌ LightGBM metrics not found at: {lgb_file}")
    print("   Run: python src/models/lightgbm_standalone.py first")

# Display comparison
if len(results) > 0:
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    for model, auc in results.items():
        print(f"   {model:10s} : {auc:.4f}")
    
    print("-"*60)
    
    if len(results) == 2:
        if results['XGBoost'] > results['LightGBM']:
            diff = results['XGBoost'] - results['LightGBM']
            print(f"   🏆 WINNER: XGBoost (+{diff:.4f})")
        elif results['LightGBM'] > results['XGBoost']:
            diff = results['LightGBM'] - results['XGBoost']
            print(f"   🏆 WINNER: LightGBM (+{diff:.4f})")
        else:
            print("   🤝 TIE! Both models performed equally")
    
    print("="*60)

# Also show training times if available
print("\n⏱️  TRAINING TIMES:")
print("-"*40)

# XGBoost training time
if xgb_file.exists():
    with open(xgb_file, 'r') as f:
        for line in f:
            if 'Training Time:' in line:
                time_str = line.split(':')[1].strip()
                print(f"   XGBoost : {time_str}")
                break

# LightGBM training time
if lgb_file.exists():
    with open(lgb_file, 'r') as f:
        for line in f:
            if 'Training Time:' in line:
                time_str = line.split(':')[1].strip()
                print(f"   LightGBM: {time_str}")
                break

print("\n✅ Comparison complete!")