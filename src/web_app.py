"""Simple Web Interface - Direct API Calls"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import httpx
import uvicorn

app = FastAPI()

# Deployed API configuration
API_URL = "https://fraud-detection-using-ml.onrender.com"
API_KEY = "prod_key_123"

# HTML template as string (no external files needed)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        input, select {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            width: 100%;
            background: #667eea;
            color: white;
            padding: 12px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        button:hover {
            background: #5a67d8;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .high-risk {
            background: #ffebee;
            border-left: 5px solid #f44336;
        }
        .medium-risk {
            background: #fff3e0;
            border-left: 5px solid #ff9800;
        }
        .low-risk {
            background: #e8f5e9;
            border-left: 5px solid #4caf50;
        }
        .loading {
            text-align: center;
            color: #667eea;
            margin-top: 20px;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 AI Fraud Detection System</h1>
        <p style="text-align: center; color: #666;">Powered by XGBoost + LightGBM Ensemble</p>
        
        <form id="fraudForm">
            <label>💰 Transaction Amount ($):</label>
            <input type="number" id="amount" step="0.01" required placeholder="Enter amount">
            
            <label>📦 Product Code:</label>
            <select id="product">
                <option value="W">W - Consumer Products</option>
                <option value="C">C - High Risk Category</option>
                <option value="R">R - Retail</option>
            </select>
            
            <label>💳 Card ID:</label>
            <input type="number" id="card_id" value="12345">
            
            <button type="submit">🔍 Check Fraud Risk</button>
        </form>
        
        <div id="loading" class="loading" style="display:none;">
            <p>Analyzing transaction with AI models...</p>
        </div>
        
        <div id="result" style="display:none;"></div>
    </div>

    <script>
        const API_URL = "https://fraud-detection-using-ml.onrender.com";
        const API_KEY = "prod_key_123";
        
        document.getElementById('fraudForm').onsubmit = async (e) => {
            e.preventDefault();
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            const amount = parseFloat(document.getElementById('amount').value);
            const product = document.getElementById('product').value;
            const card_id = parseInt(document.getElementById('card_id').value);
            
            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: {
                        'X-API-Key': API_KEY,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        TransactionAmt: amount,
                        ProductCD: product,
                        card1: card_id
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`API Error: ${response.status}`);
                }
                
                const result = await response.json();
                displayResult(result);
                
            } catch (error) {
                displayError(error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        };
        
        function displayResult(result) {
            const resultDiv = document.getElementById('result');
            const riskLevel = result.risk_level;
            
            let riskClass = '';
            let riskIcon = '';
            
            if (riskLevel === 'HIGH RISK') {
                riskClass = 'high-risk';
                riskIcon = '🔴';
            } else if (riskLevel === 'MEDIUM RISK') {
                riskClass = 'medium-risk';
                riskIcon = '🟡';
            } else {
                riskClass = 'low-risk';
                riskIcon = '🟢';
            }
            
            resultDiv.className = `result ${riskClass}`;
            resultDiv.innerHTML = `
                <h3>${riskIcon} Prediction Result</h3>
                <p><strong>Fraud Probability:</strong> 
                    <span style="font-size: 24px; font-weight: bold;">${(result.fraud_probability * 100).toFixed(1)}%</span>
                </p>
                <p><strong>Risk Level:</strong> ${riskLevel}</p>
                <p><strong>Is Fraud:</strong> ${result.is_fraud ? '⚠️ YES - Flag for review' : '✅ NO - Looks legitimate'}</p>
                <p><strong>🤖 XGBoost Score:</strong> ${(result.xgb_score * 100).toFixed(1)}%</p>
                <p><strong>🤖 LightGBM Score:</strong> ${(result.lgb_score * 100).toFixed(1)}%</p>
                <p><strong>📝 Explanation:</strong> ${result.explanation}</p>
                <p><small>🕐 ${new Date(result.timestamp).toLocaleString()}</small></p>
            `;
            resultDiv.style.display = 'block';
        }
        
        function displayError(error) {
            const resultDiv = document.getElementById('result');
            resultDiv.className = 'result error';
            resultDiv.innerHTML = `
                <h3>❌ Error</h3>
                <p>Could not connect to fraud detection API.</p>
                <p><strong>Error:</strong> ${error}</p>
                <p><small>Please try again later or contact support.</small></p>
            `;
            resultDiv.style.display = 'block';
        }
        
        // Test API connection on page load
        async function testConnection() {
            try {
                const response = await fetch(`${API_URL}/health`);
                if (response.ok) {
                    console.log('✅ API connection successful');
                }
            } catch (error) {
                console.log('⚠️ API connection failed');
            }
        }
        testConnection();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.get("/health")
async def health():
    """Check if deployed API is healthy"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_URL}/health")
            return {"status": "ok", "api_status": response.json()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 FRAUD DETECTION WEB APP")
    print("="*50)
    print(f"\n📍 Open in browser: http://127.0.0.1:8000")
    print(f"\n⚠️  Press CTRL+C to stop")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")