"""Test the Fraud Detection API"""
import requests
import json

# API endpoint
API_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print(f"Health Check: {response.json()}")

def test_single_prediction():
    """Test single transaction prediction"""
    # Example transaction
    transaction = {
        "TransactionAmt": 1500.00,
        "ProductCD": "W",
        "card1": 12345,
        "card2": 67890,
        "TransactionDT": 86400000
    }
    
    response = requests.post(f"{API_URL}/predict", json=transaction)
    print("\n" + "="*50)
    print("SINGLE TRANSACTION PREDICTION")
    print("="*50)
    print(f"Transaction: {transaction}")
    print(f"\nResult: {response.json()}")

def test_batch_predictions():
    """Test multiple transactions"""
    transactions = {
        "transactions": [
            {"TransactionAmt": 50.00, "ProductCD": "W", "card1": 11111},
            {"TransactionAmt": 5000.00, "ProductCD": "C", "card1": 22222},
            {"TransactionAmt": 25.50, "ProductCD": "R", "card1": 33333}
        ]
    }
    
    response = requests.post(f"{API_URL}/predict/batch", json=transactions)
    print("\n" + "="*50)
    print("BATCH PREDICTIONS")
    print("="*50)
    result = response.json()
    print(f"Total: {result['total_transactions']}")
    print(f"High Risk: {result['high_risk_count']}")
    for i, pred in enumerate(result['predictions']):
        print(f"  Transaction {i+1}: {pred['risk_level']} ({pred['fraud_probability']:.2f})")

def test_info():
    """Get model info"""
    response = requests.get(f"{API_URL}/info")
    print("\n" + "="*50)
    print("MODEL INFO")
    print("="*50)
    print(response.json())

if __name__ == "__main__":
    print("Testing Fraud Detection API...")
    print("Make sure the API is running (python src/deploy_api.py)")
    
    try:
        test_health()
        test_info()
        test_single_prediction()
        test_batch_predictions()
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API!")
        print("Please start the API first: python src/deploy_api.py")