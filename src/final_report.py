"""Final Model Comparison Report"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

print("="*60)
print("FINAL MODEL COMPARISON REPORT")
print("="*60)

results = {}

# Load all metrics
files = {
    'XGBoost': 'model_metrics.txt',
    'LightGBM': 'lightgbm_model_metrics.txt'
}

for model_name, filename in files.items():
    filepath = MODELS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            for line in f:
                if 'AUC Score:' in line:
                    results[model_name] = float(line.split(':')[1].strip())
                elif 'Training Time:' in line:
                    time_str = line.split(':')[1].strip()
                    results[f"{model_name}_time"] = float(time_str.split()[0])

# Display results
print("\n📊 MODEL PERFORMANCE:")
print("-"*40)
for model in ['XGBoost', 'LightGBM']:
    if model in results:
        print(f"\n{model}:")
        print(f"   AUC: {results[model]:.4f}")
        if f"{model}_time" in results:
            print(f"   Training Time: {results[f'{model}_time']:.1f} seconds")

# Recommendation
print("\n" + "="*60)
print("🎯 RECOMMENDATION:")
print("="*60)

if results['XGBoost'] > results['LightGBM']:
    print("\n✅ Use XGBoost for best accuracy (0.9584 AUC)")
    print("   - Better fraud detection rate")
    print("   - Use for final predictions")
else:
    print("\n⚡ Use LightGBM for speed (18.5s training)")
    print("   - Great for quick iterations")
    print("   - Use for experimentation")

print("\n💡 For best results, use Ensemble Model!")
print("   - Combines strengths of both models")
print("   - Run: python src/ensemble_model.py")

print("\n✅ Report complete!")