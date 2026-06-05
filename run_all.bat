@echo off
echo ========================================
echo TRAINING AND COMPARING ALL MODELS
echo ========================================

echo.
echo [1/3] Training XGBoost...
python src/models/xgboost_baseline.py

echo.
echo [2/3] Training LightGBM...
python src/models/lightgbm_standalone.py

echo.
echo [3/3] Comparing results...
python src/compare_results.py

echo.
echo Done! Press any key to exit...
pause > nul