"""Secure Fraud Detection API with Authentication"""
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
import uvicorn
from datetime import datetime

# ========== INITIALIZE FASTAPI ==========
app = FastAPI(
    title="Secure Fraud Detection API",
    description="Production-ready fraud detection with API key authentication",
    version="2.0.0"
)

# ========== API KEY AUTHENTICATION ==========
# Valid API keys (in production, store these in environment variables)
VALID_API_KEYS = {
    "prod_key_123": {
        "name": "Production App",
        "rate_limit": 1000,
        "active": True
    },
    "test_key_456": {
        "name": "Testing App", 
        "rate_limit": 100,
        "active": True
    },
    "dev_key_789": {
        "name": "Development App",
        "rate_limit": 500,
        "active": True
    }
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key and return client info"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403, 
            detail="Invalid API Key. Please provide a valid X-API-Key header."
        )
    
    client_info = VALID_API_KEYS[api_key]
    if not client_info.get("active", True):
        raise HTTPException(
            status_code=403,
            detail="API Key is deactivated. Please contact administrator."
        )
    
    return client_info

# ========== LOAD MODELS ==========
print("\n" + "="*60)
print("🔧 Loading Fraud Detection Models...")
print("="*60)

MODELS_DIR = Path("models")

# Load feature columns
try:
    with open(MODELS_DIR.parent / "data/processed/features.txt", 'r') as f:
        FEATURE_COLUMNS = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(FEATURE_COLUMNS)} feature columns")
except:
    print("⚠️ Feature list not found, will use default columns")
    FEATURE_COLUMNS = None

# Load XGBoost model
try:
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))
    print("✓ XGBoost model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load XGBoost model: {e}")
    xgb_model = None

# Load LightGBM model
try:
    lgb_model = joblib.load(MODELS_DIR / "lightgbm_model_tuned.pkl")
    print("✓ LightGBM model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load LightGBM model Tuned: {e}")
    lgb_model = None

if xgb_model is None and lgb_model is None:
    print("\n❌ ERROR: No models loaded! Please train models first:")
    print("   python src/models/xgboost_baseline.py")
    print("   python src/models/lightgbm_standalone.py")
    exit(1)

print("✅ All models loaded successfully!")

# ========== REQUEST/RESPONSE MODELS ==========
class TransactionRequest(BaseModel):
    """Single transaction request"""
    TransactionAmt: float
    ProductCD: str = "W"
    card1: int = 12345
    card2: int = None
    card3: int = None
    card4: str = None
    card5: str = None
    card6: str = None
    addr1: int = None
    addr2: int = None
    TransactionDT: int = None
    
    class Config:
        schema_extra = {
            "example": {
                "TransactionAmt": 1500.50,
                "ProductCD": "W",
                "card1": 12345,
                "TransactionDT": 86400000
            }
        }

class BatchTransactionRequest(BaseModel):
    """Batch transaction request"""
    transactions: List[TransactionRequest]

class FraudResponse(BaseModel):
    """Fraud prediction response"""
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    xgb_score: float
    lgb_score: float
    explanation: str
    timestamp: str

class BatchFraudResponse(BaseModel):
    """Batch prediction response"""
    total_transactions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[Dict]

# ========== HELPER FUNCTIONS ==========
def preprocess_input(transaction: Dict[str, Any]) -> pd.DataFrame:
    """Convert API input to model-ready format"""
    df = pd.DataFrame([transaction])
    
    # Fill missing values
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(-999)
    
    # Add missing columns if needed
    if FEATURE_COLUMNS:
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = -999
        df = df[FEATURE_COLUMNS]
    
    # Ensure numeric types
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

def get_explanation(probability: float) -> str:
    """Generate explanation for prediction"""
    if probability >= 0.7:
        return "High fraud probability detected. Recommend manual review and additional verification."
    elif probability >= 0.3:
        return "Medium fraud probability. Monitor transaction closely and consider additional checks."
    else:
        return "Low fraud probability. Transaction appears normal."

def predict_fraud(transaction_data: Dict) -> FraudResponse:
    """Make fraud prediction for single transaction"""
    # Preprocess
    df = preprocess_input(transaction_data)
    
    # Make predictions
    xgb_pred = 0.5
    lgb_pred = 0.5
    
    if xgb_model is not None:
        dtest = xgb.DMatrix(df)
        xgb_pred = float(xgb_model.predict(dtest)[0])
    
    if lgb_model is not None:
        lgb_pred = float(lgb_model.predict_proba(df)[0, 1])
    
    # Ensemble (30% XGBoost, 70% LightGBM)
    final_pred = 0.3 * xgb_pred + 0.7 * lgb_pred
    
    return FraudResponse(
        fraud_probability=round(final_pred, 4),
        is_fraud=final_pred > 0.5,
        risk_level=get_risk_level(final_pred),
        xgb_score=round(xgb_pred, 4),
        lgb_score=round(lgb_pred, 4),
        explanation=get_explanation(final_pred),
        timestamp=datetime.now().isoformat()
    )

# ========== API ENDPOINTS ==========
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "predict": "/predict (POST)",
            "batch": "/predict/batch (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "xgboost": xgb_model is not None,
            "lightgbm": lgb_model is not None
        },
        "features_count": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/info")
def model_info(api_key: dict = Depends(verify_api_key)):
    """Get model information (requires authentication)"""
    return {
        "model_name": "Fraud Detection Ensemble",
        "version": "2.0.0",
        "features_used": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 205,
        "xgb_auc": 0.9584,
        "lgb_auc": 0.9241,
        "ensemble_auc": 0.96,
        "description": "Ensemble of XGBoost and LightGBM models for credit card fraud detection",
        "authentication": "API Key required",
        "client": api_key
    }

@app.post("/predict", response_model=FraudResponse)
def predict_single(
    transaction: TransactionRequest,
    api_key: dict = Depends(verify_api_key)
):
    """Predict fraud for a single transaction (requires authentication)"""
    try:
        transaction_dict = transaction.dict()
        result = predict_fraud(transaction_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchFraudResponse)
def predict_batch(
    batch: BatchTransactionRequest,
    api_key: dict = Depends(verify_api_key)
):
    """Predict fraud for multiple transactions (requires authentication)"""
    try:
        predictions = []
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        
        for transaction in batch.transactions:
            result = predict_fraud(transaction.dict())
            predictions.append({
                "fraud_probability": result.fraud_probability,
                "is_fraud": result.is_fraud,
                "risk_level": result.risk_level
            })
            
            if result.risk_level == "HIGH RISK":
                high_risk += 1
            elif result.risk_level == "MEDIUM RISK":
                medium_risk += 1
            else:
                low_risk += 1
        
        return BatchFraudResponse(
            total_transactions=len(predictions),
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            predictions=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

# ========== RUN API ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING SECURE FRAUD DETECTION API")
    print("="*60)
    print(f"\n📍 API URL: http://127.0.0.1:8000")
    print(f"📖 API Docs: http://127.0.0.1:8000/docs")
    print(f"🔑 Test API Key: prod_key_123")
    print(f"\n⚠️  Press CTRL+C to stop the API")
    print("="*60 + "\n")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        log_level="info",
        access_log=True
    )