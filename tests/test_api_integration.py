"""
Integration tests for StockFlowML API with Feast Feature Store

Tests both manual feature input and Feast-based ticker prediction.
"""
import sys
from pathlib import Path
import pytest
import requests
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# API base URL
API_URL = "http://localhost:8000"


def test_api_health():
    """Test API health check endpoint."""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    print(f"✓ Health check passed: {data}")


def test_manual_prediction():
    """Test prediction with manual features."""
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
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "processing_time_ms" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    
    print(f"✓ Manual prediction: {data['prediction']} (prob: {data['probability']:.3f}, latency: {data['processing_time_ms']:.2f}ms)")


def test_ticker_prediction():
    """Test prediction with ticker (Feast feature store)."""
    payload = {
        "ticker": "BBCA.JK",
        "timestamp": None  # Use latest
    }
    
    response = requests.post(f"{API_URL}/predict/ticker", json=payload)
    
    # This might fail if Feast is not initialized or features not materialized
    if response.status_code == 503:
        print("⚠ Feast not available - skipping ticker prediction test")
        pytest.skip("Feast feature store not available")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    
    print(f"✓ Ticker prediction: {data['prediction']} (prob: {data['probability']:.3f}, latency: {data['processing_time_ms']:.2f}ms)")


def test_invalid_input():
    """Test API error handling with invalid input."""
    payload = {
        "Open": 4500.0,
        # Missing required fields
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_latency():
    """Test API latency is within acceptable range."""
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
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    data = response.json()
    
    # Latency should be < 100ms for production
    assert data["processing_time_ms"] < 100, f"Latency too high: {data['processing_time_ms']}ms"
    print(f"✓ Latency check passed: {data['processing_time_ms']:.2f}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("StockFlowML API Integration Tests")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print()
    
    try:
        # Check if API is running
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print("✗ API is not healthy. Please start the API server first:")
            print("  python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Please start the API server first:")
        print("  python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Run tests
    print("Running tests...\n")
    
    test_api_health()
    test_manual_prediction()
    test_ticker_prediction()
    test_invalid_input()
    test_latency()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
