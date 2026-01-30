"""
Tests for model training and prediction.
"""

import pytest
import numpy as np
from src.models.train import StockTrendModel
from src.utils.config import Config


def test_model_initialization():
    """Test model initialization."""
    model = StockTrendModel(model_type='logistic')
    assert model.model_type == 'logistic'
    assert model.model is None


def test_model_training():
    """Test basic model training."""
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    model = StockTrendModel(model_type='logistic')
    model.feature_names = [f'feature_{i}' for i in range(5)]
    model.train(X_train, y_train)
    
    # Check model is trained
    assert model.model is not None
    assert model.scaler is not None


def test_model_prediction():
    """Test model prediction."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 5)
    
    model = StockTrendModel(model_type='logistic')
    model.feature_names = [f'feature_{i}' for i in range(5)]
    model.train(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Check predictions
    assert len(predictions) == 20
    assert all(pred in [0, 1] for pred in predictions)


def test_model_predict_proba():
    """Test probability predictions."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 5)
    
    model = StockTrendModel(model_type='logistic')
    model.feature_names = [f'feature_{i}' for i in range(5)]
    model.train(X_train, y_train)
    
    probas = model.predict_proba(X_test)
    
    # Check probabilities
    assert probas.shape == (20, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
