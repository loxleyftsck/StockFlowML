import requests
import json
import time

def test_api():
    print("Testing API Integration...")
    base_url = "http://localhost:8000"
    
    # 1. Health Check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200 and response.json()["status"] == "ok":
            print("✅ Health Check Passed")
        else:
            print("❌ Health Check Failed")
            return
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 2. Prediction
    payload = {
        "Open": 4500.0,
        "High": 4550.0,
        "Low": 4480.0,
        "Close": 4520.0,
        "Volume": 10000000.0,
        "returns": 0.005,
        "ma_5": 4480.0,
        "ma_10": 4470.0,
        "ma_20": 4450.0,
        "volatility_5": 0.012,
        "volatility_10": 0.013,
        "volatility_20": 0.015
    }
    
    try:
        start = time.time()
        response = requests.post(f"{base_url}/predict", json=payload)
        latency = (time.time() - start) * 1000
        
        print(f"\nPrediction: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"Latency: {latency:.2f}ms")
        
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data and "probability" in data:
                print("✅ Prediction Passed")
            else:
                print("❌ Invalid Response Schema")
        else:
            print("❌ Prediction Failed")
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_api()
