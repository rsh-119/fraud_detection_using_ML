"""Fraud Detection API - Deploy Model as Web Service"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any
import uvicorn

# ========== INITIALIZE API ==========
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using XGBoost & LightGBM",
    version="1.0.0"
)

# ========== LOAD MODELS ==========
print("Loading models...")
MODELS_DIR = Path("models")

# Load XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))
print("✓ XGBoost model loaded")

# Load LightGBM model
lgb_model = joblib.load(MODELS_DIR / "lightgbm_model.pkl")
print("✓ LightGBM model loaded")

# Load feature list (to ensure input has correct columns)
with open(MODELS_DIR.parent / "data/processed/features.txt", 'r') as f:
    FEATURE_COLUMNS = [line.strip() for line in f.readlines()]
print(f"✓ Loaded {len(FEATURE_COLUMNS)} features")

# ========== DEFINE REQUEST SCHEMA ==========
class TransactionRequest(BaseModel):
    """Expected input format for transaction data"""
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: int = None
    card3: int = None
    card4: str = None
    card5: str = None
    card6: str = None
    addr1: int = None
    addr2: int = None
    dist1: float = None
    TransactionDT: int = None
    # Add more features as needed
    
    class Config:
        schema_extra = {
            "example": {
                "TransactionAmt": 250.50,
                "ProductCD": "W",
                "card1": 12345,
                "card2": 67890,
                "TransactionDT": 86400000
            }
        }

class BatchTransactionRequest(BaseModel):
    """Batch prediction for multiple transactions"""
    transactions: list[TransactionRequest]

class FraudResponse(BaseModel):
    """API Response format"""
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    xgb_score: float
    lgb_score: float
    explanation: str

# ========== HELPER FUNCTIONS ==========
def preprocess_input(transaction: Dict[str, Any]) -> pd.DataFrame:
    """Convert API input to model-ready format"""
    # Create DataFrame from input
    df = pd.DataFrame([transaction])
    
    # Fill missing values (same as training)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = -999  # Default missing value
    
    # Ensure all columns exist and in correct order
    df = df[FEATURE_COLUMNS]
    
    # Convert to numeric (handle strings)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999)
    
    return df

def get_risk_level(probability: float) -> str:
    """Categorize risk level"""
    if probability >= 0.7:
        return "HIGH RISK"
    elif probability >= 0.3:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

# ========== API ENDPOINTS ==========
@app.get("/")
def home():
    """Welcome endpoint"""
    return {
        "message": "Fraud Detection API is running!",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single transaction prediction",
            "/predict/batch": "POST - Batch transaction prediction",
            "/health": "GET - Check API health",
            "/info": "GET - Model information"
        }
    }

@app.get("/health")
def health_check():
    """Check if API and models are working"""
    return {
        "status": "healthy",
        "models_loaded": {
            "xgboost": True,
            "lightgbm": True
        },
        "features_count": len(FEATURE_COLUMNS)
    }

@app.get("/info")
def model_info():
    """Get model information"""
    return {
        "model_name": "Fraud Detection Ensemble",
        "version": "1.0.0",
        "features_used": len(FEATURE_COLUMNS),
        "xgb_auc": 0.9584,
        "lgb_auc": 0.9241,
        "ensemble_auc": 0.96,
        "description": "Ensemble of XGBoost and LightGBM for fraud detection"
    }

@app.post("/predict", response_model=FraudResponse)
def predict_fraud(transaction: TransactionRequest):
    """Predict fraud probability for a single transaction"""
    try:
        # Convert to dict and preprocess
        transaction_dict = transaction.dict()
        df = preprocess_input(transaction_dict)
        
        # Make predictions
        dtest = xgb.DMatrix(df)
        xgb_pred = float(xgb_model.predict(dtest)[0])
        lgb_pred = float(lgb_model.predict_proba(df)[0, 1])
        
        # Ensemble (70% XGBoost, 30% LightGBM - based on your results)
        final_pred = (0.7 * xgb_pred + 0.3 * lgb_pred)
        
        # Get risk level
        risk_level = get_risk_level(final_pred)
        
        # Generate explanation
        if final_pred >= 0.7:
            explanation = "High fraud probability detected. Recommend manual review."
        elif final_pred >= 0.3:
            explanation = "Medium fraud probability. Monitor transaction."
        else:
            explanation = "Low fraud probability. Transaction appears normal."
        
        return FraudResponse(
            fraud_probability=round(final_pred, 4),
            is_fraud=final_pred > 0.5,
            risk_level=risk_level,
            xgb_score=round(xgb_pred, 4),
            lgb_score=round(lgb_pred, 4),
            explanation=explanation
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(transactions: BatchTransactionRequest):
    """Predict fraud for multiple transactions"""
    try:
        results = []
        
        for transaction in transactions.transactions:
            transaction_dict = transaction.dict()
            df = preprocess_input(transaction_dict)
            
            # Make predictions
            dtest = xgb.DMatrix(df)
            xgb_pred = float(xgb_model.predict(dtest)[0])
            lgb_pred = float(lgb_model.predict_proba(df)[0, 1])
            final_pred = 0.7 * xgb_pred + 0.3 * lgb_pred
            
            results.append({
                "fraud_probability": round(final_pred, 4),
                "is_fraud": final_pred > 0.5,
                "risk_level": get_risk_level(final_pred)
            })
        
        return {
            "total_transactions": len(results),
            "high_risk_count": sum(1 for r in results if r["risk_level"] == "HIGH RISK"),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

# ========== RUN API ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING FRAUD DETECTION API")
    print("="*60)
    print(f"\n📍 API will be available at: http://127.0.0.1:8000")
    print(f"📖 API Documentation: http://127.0.0.1:8000/docs")
    print(f"📊 Interactive API: http://127.0.0.1:8000/redoc")
    print("\n⚠️  Press CTRL+C to stop the API")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")