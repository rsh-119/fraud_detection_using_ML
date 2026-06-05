"""Complete Fraud Detection Web App - Single File Solution"""
from flask import Flask, render_template_string, request, jsonify
import requests
import json

app = Flask(__name__)

# Deployed API configuration
API_URL = "https://fraud-detection-using-ml.onrender.com"
API_KEY = "prod_key_123"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-weight: bold;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 15px;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .high-risk {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left: 5px solid #f44336;
        }
        
        .medium-risk {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-left: 5px solid #ff9800;
        }
        
        .low-risk {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 5px solid #4caf50;
        }
        
        .probability {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        
        .risk-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        
        .badge-high { background: #f44336; color: white; }
        .badge-medium { background: #ff9800; color: white; }
        .badge-low { background: #4caf50; color: white; }
        
        .scores {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(0,0,0,0.1);
        }
        
        .score {
            text-align: center;
            flex: 1;
        }
        
        .score-value {
            font-size: 20px;
            font-weight: bold;
        }
        
        .explanation {
            margin-top: 15px;
            padding: 10px;
            background: rgba(0,0,0,0.05);
            border-radius: 10px;
            font-size: 14px;
        }
        
        .timestamp {
            margin-top: 15px;
            font-size: 12px;
            color: #999;
            text-align: center;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 5px solid #c62828;
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }
            .probability {
                font-size: 36px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 AI Fraud Detection System</h1>
        <div class="subtitle">Powered by XGBoost + LightGBM Ensemble (95.8% AUC)</div>
        
        <form id="fraudForm">
            <div class="form-group">
                <label>💰 Transaction Amount ($)</label>
                <input type="number" id="amount" step="0.01" required placeholder="Enter amount" value="12000">
            </div>
            
            <div class="form-group">
                <label>📦 Product Code</label>
                <select id="product">
                    <option value="W">W - Consumer Products</option>
                    <option value="C" selected>C - High Risk Category</option>
                    <option value="R">R - Retail</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>💳 Card ID</label>
                <input type="number" id="card_id" value="51255">
            </div>
            
            <button type="submit">🔍 Check Fraud Risk</button>
        </form>
        
        <div id="loading" style="display:none;">
            <div class="loading">
                <div>🔄 Analyzing transaction with AI models...</div>
            </div>
        </div>
        
        <div id="result" style="display:none;"></div>
    </div>

    <script>
        document.getElementById('fraudForm').onsubmit = async (e) => {
            e.preventDefault();
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            const amount = parseFloat(document.getElementById('amount').value);
            const product = document.getElementById('product').value;
            const card_id = parseInt(document.getElementById('card_id').value);
            
            // Call our backend (not the API directly)
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    amount: amount,
                    product: product,
                    card_id: card_id
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                displayError(data.error);
            } else {
                displayResult(data);
            }
            
            document.getElementById('loading').style.display = 'none';
        };
        
        function displayResult(result) {
            const resultDiv = document.getElementById('result');
            const riskLevel = result.risk_level;
            
            let riskClass = '';
            if (riskLevel === 'HIGH RISK') riskClass = 'high-risk';
            else if (riskLevel === 'MEDIUM RISK') riskClass = 'medium-risk';
            else riskClass = 'low-risk';
            
            resultDiv.className = `result ${riskClass}`;
            resultDiv.innerHTML = `
                <div class="probability">
                    ${(result.fraud_probability * 100).toFixed(1)}%
                </div>
                <div style="text-align: center; margin-bottom: 15px;">
                    <span class="risk-badge badge-${riskLevel === 'HIGH RISK' ? 'high' : riskLevel === 'MEDIUM RISK' ? 'medium' : 'low'}">
                        ${riskLevel}
                    </span>
                    <span style="margin-left: 10px;">
                        ${result.is_fraud ? '⚠️ FRAUD ALERT' : '✅ LEGITIMATE'}
                    </span>
                </div>
                <div class="scores">
                    <div class="score">
                        <div>XGBoost</div>
                        <div class="score-value">${(result.xgb_score * 100).toFixed(1)}%</div>
                    </div>
                    <div class="score">
                        <div>LightGBM</div>
                        <div class="score-value">${(result.lgb_score * 100).toFixed(1)}%</div>
                    </div>
                </div>
                <div class="explanation">
                    📝 ${result.explanation}
                </div>
                <div class="timestamp">
                    🕐 ${new Date(result.timestamp).toLocaleString()}
                </div>
            `;
            resultDiv.style.display = 'block';
        }
        
        function displayError(error) {
            const resultDiv = document.getElementById('result');
            resultDiv.className = 'result error';
            resultDiv.innerHTML = `
                <h3>❌ Error</h3>
                <p>${error}</p>
                <p><small>Please try again later.</small></p>
            `;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare request for deployed API
        api_payload = {
            "TransactionAmt": data['amount'],
            "ProductCD": data['product'],
            "card1": data['card_id']
        }
        
        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        
        # Call deployed API
        response = requests.post(
            f"{API_URL}/predict",
            json=api_payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({"error": f"API Error: {response.status_code}"}), 500
        
        result = response.json()
        return jsonify(result)
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "API timeout - please try again"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return jsonify({"status": "ok", "api": response.json()})
    except:
        return jsonify({"status": "error", "api": "unreachable"})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 FRAUD DETECTION WEB APP")
    print("="*50)
    print(f"\n📍 Open in browser: http://127.0.0.1:5000")
    print(f"\n⚠️  Press CTRL+C to stop")
    print("="*50 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=True)