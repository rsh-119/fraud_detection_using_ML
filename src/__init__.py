"""Models package - Import all available models"""
from .xgboost_baseline import XGBoostModel
from .lightgbm_model import LightGBMModel
from .base_model import BaseModel

# Dictionary of available models
AVAILABLE_MODELS = {
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel,
}

def get_model(model_name, **kwargs):
    """Factory function to get model instance"""
    if model_name.lower() not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_name.lower()](**kwargs)