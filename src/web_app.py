"""Simple Web Interface for Fraud Detection"""
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import xgboost as xgb

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load models
MODELS_DIR = Path("models")
xgb_model = xgb.Booster()
xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))
lgb_model = joblib.load(MODELS_DIR / "lightgbm_model.pkl")

class Transaction(BaseModel):
    amount: float
    product: str = "W"
    card_id: int = 12345

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(transaction: Transaction):
    # Create DataFrame
    df = pd.DataFrame([{
        'TransactionAmt': transaction.amount,
        'ProductCD': transaction.product,
        'card1': transaction.card_id,
        # Add other default features
    }])
    
    # Fill missing columns
    for col in range(205):  # Your feature count
        col_name = f"V{col}" if col > 0 else f"V{col}"
        if col_name not in df.columns:
            df[col_name] = -999
    
    # Predict
    dtest = xgb.DMatrix(df)
    xgb_pred = float(xgb_model.predict(dtest)[0])
    lgb_pred = float(lgb_model.predict_proba(df)[0, 1])
    final_pred = 0.7 * xgb_pred + 0.3 * lgb_pred
    
    return {
        "fraud_probability": round(final_pred, 4),
        "is_fraud": final_pred > 0.5,
        "risk_level": "HIGH" if final_pred > 0.7 else "MEDIUM" if final_pred > 0.3 else "LOW",
        "xgb_score": round(xgb_pred, 4),
        "lgb_score": round(lgb_pred, 4)
    }