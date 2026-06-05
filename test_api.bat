@echo off
echo Testing Fraud Detection API...
echo.
curl -X POST https://fraud-detection-using-ml.onrender.com/predict -H "X-API-Key: prod_key_123" -H "Content-Type: application/json" -d "{\"TransactionAmt\":1500,\"ProductCD\":\"W\",\"card1\":12345}"
echo.
pause