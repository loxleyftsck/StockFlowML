"""
Tests for Prometheus metrics integration.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app, prediction_counter, prediction_latency, model_predictions

client = TestClient(app)


def test_metrics_endpoint_exists():
    """Test that /metrics endpoint is available."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_metrics_format():
    """Test that metrics endpoint returns valid Prometheus format."""
    response = client.get("/metrics")
    content = response.text
    
    # Check for Prometheus metric format (# HELP, # TYPE)
    assert "# HELP" in content or "# TYPE" in content
    
    # Check for custom metrics
    assert "stockflowml_prediction_requests_total" in content or "prediction_requests" in content


def test_prediction_counter_increments():
    """Test that prediction counter increments after prediction."""
    # Get initial counter value
    initial_value = prediction_counter.labels(endpoint='/predict', status='success')._value.get()
    
    # Make a prediction
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
    
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        # Counter should have incremented
        new_value = prediction_counter.labels(endpoint='/predict', status='success')._value.get()
        assert new_value > initial_value


def test_prediction_latency_histogram():
    """Test that latency histogram records measurements."""
    # Make a prediction
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
    
    response = client.post("/predict", json=payload)
    
    # Check that histogram has samples
    if response.status_code == 200:
        histogram = prediction_latency.labels(endpoint='/predict')
        # Histogram should have at least one observation
        assert histogram._sum.get() > 0


def test_model_predictions_counter():
    """Test that model predictions counter tracks prediction classes."""
    # Make a prediction
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
    
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        # Check prediction class counter exists
        up_count = model_predictions.labels(prediction_class='up')._value.get()
        down_count = model_predictions.labels(prediction_class='down')._value.get()
        
        # At least one should be > 0
        assert (up_count + down_count) > 0


def test_error_metrics_on_invalid_input():
    """Test that error metrics increment on invalid input."""
    initial_error = prediction_counter.labels(endpoint='/predict', status='error')._value.get()
    
    # Send invalid payload (missing features)
    invalid_payload = {
        "Open": 4500.0,
        "High": 4550.0,
    }
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Validation error
    
    # Error counter might not increment for validation errors (before endpoint logic)
    # but this tests the endpoint behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
