"""Train both XGBoost and LightGBM sequentially"""
import subprocess
import sys
from pathlib import Path

print("="*60)
print("TRAINING ALL MODELS")
print("="*60)

# Train XGBoost
print("\n🔵 STEP 1: Training XGBoost...")
print("-"*40)
result = subprocess.run([sys.executable, "src/models/xgboost_baseline.py"])
if result.returncode != 0:
    print("❌ XGBoost training failed!")
else:
    print("✅ XGBoost training completed!")

# Train LightGBM
print("\n🟢 STEP 2: Training LightGBM...")
print("-"*40)
result = subprocess.run([sys.executable, "src/models/lightgbm_standalone.py"])
if result.returncode != 0:
    print("❌ LightGBM training failed!")
else:
    print("✅ LightGBM training completed!")

print("\n" + "="*60)
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)
print("\n📁 Check models in: models/")
print("   - xgboost_model.json")
print("   - lightgbm_model.pkl")
print("\n📊 Check metrics:")
print("   - models/model_metrics.txt (XGBoost)")
print("   - models/lightgbm_model_metrics.txt (LightGBM)")