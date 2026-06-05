"""Complete API Testing Suite - Test all endpoints"""
import requests
import json
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://127.0.0.1:8000"
API_KEY = "prod_key_123"  # Valid API key for testing

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}📌 {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️ {msg}{Colors.RESET}")

def print_section(title):
    print("\n" + "="*60)
    print(f"{Colors.BLUE}{title}{Colors.RESET}")
    print("="*60)

# ========== 1. TEST HEALTH ENDPOINT ==========
def test_health():
    print_section("1. Testing Health Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed! (Status: {response.status_code})")
            print(f"   Status: {data['status']}")
            print(f"   Models Loaded: {data['models_loaded']}")
            print(f"   Features Count: {data['features_count']}")
            return True
        else:
            print_error(f"Health check failed! Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cannot connect to API: {e}")
        print_info("Make sure API is running: python src/secure_api.py")
        return False

# ========== 2. TEST INFO ENDPOINT ==========
def test_info():
    print_section("2. Testing Info Endpoint")
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{BASE_URL}/info", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Info endpoint working! (Status: {response.status_code})")
            print(f"   Model: {data['model_name']}")
            print(f"   Version: {data['version']}")
            print(f"   Features: {data['features_used']}")
            print(f"   XGBoost AUC: {data['xgb_auc']}")
            print(f"   LightGBM AUC: {data['lgb_auc']}")
            return True
        else:
            print_error(f"Info endpoint failed! Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

# ========== 3. TEST PREDICT ENDPOINT (Valid API Key) ==========
def test_predict_valid_key():
    print_section("3. Testing Predict Endpoint (Valid API Key)")
    
    transaction = {
        "TransactionAmt": 1500.00,
        "ProductCD": "W",
        "card1": 12345,
        "card2": 67890,
        "TransactionDT": 86400000
    }
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Prediction successful! (Status: {response.status_code})")
            print(f"   Fraud Probability: {data['fraud_probability']}")
            print(f"   Is Fraud: {data['is_fraud']}")
            print(f"   Risk Level: {data['risk_level']}")
            print(f"   XGBoost Score: {data['xgb_score']}")
            print(f"   LightGBM Score: {data['lgb_score']}")
            print(f"   Explanation: {data['explanation']}")
            return True
        else:
            print_error(f"Prediction failed! Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

# ========== 4. TEST PREDICT ENDPOINT (Invalid API Key) ==========
def test_predict_invalid_key():
    print_section("4. Testing Predict Endpoint (Invalid API Key)")
    
    transaction = {"TransactionAmt": 500.00, "ProductCD": "C", "card1": 99999}
    headers = {"X-API-Key": "wrong_key_123"}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction,
            headers=headers
        )
        
        if response.status_code == 403:
            print_success(f"Invalid key correctly rejected (Status: {response.status_code})")
            return True
        else:
            print_warning(f"Expected 403, got {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

# ========== 5. TEST PREDICT ENDPOINT (No API Key) ==========
def test_predict_no_key():
    print_section("5. Testing Predict Endpoint (No API Key)")
    
    transaction = {"TransactionAmt": 500.00, "ProductCD": "C", "card1": 99999}
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=transaction)
        
        # FastAPI returns 401 for missing API key
        if response.status_code == 401 or response.status_code == 403:
            print_success(f"Security working! Request without key rejected (Status: {response.status_code})")
            return True
        else:
            print_warning(f"Expected 401 or 403, got {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

# ========== 6. TEST BATCH PREDICTIONS ==========
def test_batch_predictions():
    print_section("6. Testing Batch Predictions")
    
    transactions = {
        "transactions": [
            {"TransactionAmt": 50.00, "ProductCD": "W", "card1": 11111},
            {"TransactionAmt": 5000.00, "ProductCD": "C", "card1": 22222},
            {"TransactionAmt": 25.50, "ProductCD": "R", "card1": 33333},
            {"TransactionAmt": 15000.00, "ProductCD": "W", "card1": 44444},
            {"TransactionAmt": 100.00, "ProductCD": "C", "card1": 55555}
        ]
    }
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=transactions,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Batch prediction successful! (Status: {response.status_code})")
            print(f"   Total Transactions: {data['total_transactions']}")
            print(f"   High Risk Count: {data['high_risk_count']}")
            print(f"\n   Detailed Results:")
            for i, pred in enumerate(data['predictions'], 1):
                risk_symbol = "🔴" if pred['risk_level'] == "HIGH RISK" else "🟡" if pred['risk_level'] == "MEDIUM RISK" else "🟢"
                print(f"   {risk_symbol} Transaction {i}: {pred['risk_level']} ({pred['fraud_probability']:.2f})")
            return True
        else:
            print_error(f"Batch prediction failed! Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

# ========== 7. TEST PERFORMANCE (Response Time) ==========
def test_performance():
    print_section("7. Testing API Performance")
    
    transaction = {"TransactionAmt": 1000.00, "ProductCD": "W", "card1": 12345}
    headers = {"X-API-Key": API_KEY}
    
    response_times = []
    num_requests = 10
    
    print_info(f"Sending {num_requests} requests to measure response time...")
    
    for i in range(num_requests):
        start = time.time()
        response = requests.post(f"{BASE_URL}/predict", json=transaction, headers=headers)
        end = time.time()
        
        if response.status_code == 200:
            response_times.append((end - start) * 1000)
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print_success(f"Performance Test Complete:")
        print(f"   Average Response Time: {avg_time:.2f} ms")
        print(f"   Fastest Response: {min_time:.2f} ms")
        print(f"   Slowest Response: {max_time:.2f} ms")
        print(f"   Success Rate: {len(response_times)}/{num_requests}")
        
        if avg_time < 100:
            print_success("   Performance: Excellent (<100ms)")
        elif avg_time < 500:
            print_info("   Performance: Good (<500ms)")
        else:
            print_warning("   Performance: Slow (>500ms)")
        
        return True
    else:
        print_error("Performance test failed - no successful requests")
        return False

# ========== 8. TEST EDGE CASES ==========
def test_edge_cases():
    print_section("8. Testing Edge Cases")
    
    test_cases = [
        {"name": "Very Small Amount", "data": {"TransactionAmt": 0.01, "ProductCD": "W", "card1": 11111}},
        {"name": "Very Large Amount", "data": {"TransactionAmt": 100000, "ProductCD": "C", "card1": 22222}},
        {"name": "Missing Optional Fields", "data": {"TransactionAmt": 500, "ProductCD": "R"}},
        {"name": "Unknown Product", "data": {"TransactionAmt": 1000, "ProductCD": "X", "card1": 33333}},
    ]
    
    headers = {"X-API-Key": API_KEY}
    passed = 0
    
    for test in test_cases:
        try:
            response = requests.post(f"{BASE_URL}/predict", json=test['data'], headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"{test['name']}: {data['fraud_probability']:.2f} probability")
                passed += 1
            else:
                print_error(f"{test['name']}: Failed (Status: {response.status_code})")
        except Exception as e:
            print_error(f"{test['name']}: Error - {e}")
    
    print(f"\n   Edge Cases Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)

# ========== 9. TEST API DOCS ==========
def test_docs():
    print_section("9. Testing API Documentation")
    
    docs_urls = [
        ("Swagger UI", "/docs"),
        ("ReDoc", "/redoc"),
        ("OpenAPI JSON", "/openapi.json")
    ]
    
    all_working = True
    
    for name, url in docs_urls:
        try:
            response = requests.get(f"{BASE_URL}{url}")
            if response.status_code == 200:
                print_success(f"{name} available at: {BASE_URL}{url}")
            else:
                print_error(f"{name} failed (Status: {response.status_code})")
                all_working = False
        except Exception as e:
            print_error(f"{name} error: {e}")
            all_working = False
    
    return all_working

# ========== GENERATE REPORT ==========
def generate_report(results):
    print_section("FINAL TEST REPORT")
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"\n📊 Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    test_names = [
        "Health Endpoint",
        "Info Endpoint",
        "Predict (Valid Key)",
        "Predict (Invalid Key)",
        "Predict (No Key)",
        "Batch Predictions",
        "Performance",
        "Edge Cases",
        "API Documentation"
    ]
    
    for name, result in zip(test_names, results):
        status = f"{Colors.GREEN}✓ PASSED{Colors.RESET}" if result else f"{Colors.RED}✗ FAILED{Colors.RESET}"
        print(f"   {status} - {name}")
    
    if passed_tests == total_tests:
        print(f"\n{Colors.GREEN}{'='*60}{Colors.RESET}")
        print(f"{Colors.GREEN}🎉 ALL TESTS PASSED! API IS PRODUCTION-READY! 🎉{Colors.RESET}")
        print(f"{Colors.GREEN}{'='*60}{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}{'='*60}{Colors.RESET}")
        print(f"{Colors.YELLOW}⚠️ Some tests failed. Please check the errors above.{Colors.RESET}")
        print(f"{Colors.YELLOW}{'='*60}{Colors.RESET}")

# ========== MAIN ==========
if __name__ == "__main__":
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}🚀 FRAUD DETECTION API TEST SUITE{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"\nTesting API at: {BASE_URL}")
    print(f"Using API Key: {API_KEY}")
    
    # Check if API is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except:
        print_error("\n⚠️ API is not running!")
        print_info("Please start the API first:")
        print("   python src/secure_api.py")
        exit(1)
    
    # Run all tests
    results = []
    results.append(test_health())
    results.append(test_info())
    results.append(test_predict_valid_key())
    results.append(test_predict_invalid_key())
    results.append(test_predict_no_key())
    results.append(test_batch_predictions())
    results.append(test_performance())
    results.append(test_edge_cases())
    results.append(test_docs())
    
    # Generate report
    generate_report(results)