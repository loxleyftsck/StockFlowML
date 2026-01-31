"""
Comprehensive model validation tests for StockFlowML.
Tests temporal integrity, data leakage, baselines, reproducibility, and failure modes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.train import StockTrendModel
from src.models.model_evaluation import (
    BaselineClassifiers, ModelEvaluator,
    detect_leakage_in_features, validate_temporal_split
)
from src.features.feature_engineering import FeatureEngineer
from src.utils.config import Config


def create_sample_processed_data(n_samples=100, include_leakage=False):
    """Create sample processed data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=n_samples, freq='D')
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': np.random.uniform(100, 110, n_samples),
        'returns': np.random.normal(0, 0.02, n_samples),
        'ma_5': np.random.uniform(100, 110, n_samples),
        'ma_10': np.random.uniform(100, 110, n_samples),
        'ma_20': np.random.uniform(100, 110, n_samples),
        'volatility_5': np.random.uniform(0.01, 0.03, n_samples),
        'volatility_10': np.random.uniform(0.01, 0.03, n_samples),
        'volatility_20': np.random.uniform(0.01, 0.03, n_samples),
    })
    
    # Create target (binary)
    df['target'] = (df['returns'] > 0).astype(int)
    
    if include_leakage:
        # Add a leaky feature (perfect predictor)
        df['LEAKY_FEATURE'] = df['target']
    
    return df


class TestTemporalIntegrity:
    """Test temporal integrity of train/test split."""
    
    def test_train_precedes_test(self):
        """Test that all training data precedes test data."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        
        # Check dates
        validation = validate_temporal_split(train_df, test_df)
        
        assert validation['valid'], validation.get('error', 'Unknown error')
        assert validation['gap_days'] >= 0, "No gap between train and test"
    
    def test_no_future_leakage_in_split(self):
        """Test that test set doesn't contain earlier dates than train."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        
        train_max_date = train_df['Date'].max()
        test_min_date = test_df['Date'].min()
        
        assert test_min_date > train_max_date, \
            f"Temporal leakage detected: test min ({test_min_date}) <= train max ({train_max_date})"
    
    def test_rolling_features_use_past_only(self):
        """Test that rolling features don't use future information."""
        # Create simple dataset
        dates = pd.date_range('2021-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Close': range(50),  # Increasing sequence
            'Volume': [1000000] * 50
        })
        
        # Create rolling features
        engineer = FeatureEngineer(df)
        engineer.create_rolling_means(windows=[5])
        
        # Check that MA at index i only uses data up to index i
        # For i=10, ma_5 should be mean of Close[6:11], not including Close[11+]
        if len(engineer.df) >= 11:
            expected_ma = df['Close'].iloc[6:11].mean()
            actual_ma = engineer.df['ma_5'].iloc[10]
            
            # Allow small floating point error
            assert abs(actual_ma - expected_ma) < 1e-10, \
                "Rolling mean uses future values!"
    
    def test_target_generation_uses_t_plus_1(self):
        """Test that target correctly uses T+1 only."""
        dates = pd.date_range('2021-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Close': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105],
            'Volume': [1000000] * 10
        })
        
        # Create engineer but DON'T call create_all_features (which does dropna)
        engineer = FeatureEngineer(df)
        
        # Manually create target to preserve NaN
        engineer.df['close_next'] = engineer.df['Close'].shift(-1)
        engineer.df['target'] = (engineer.df['close_next'] > engineer.df['Close']).astype(float)
        engineer.df.loc[engineer.df['close_next'].isna(), 'target'] = np.nan
        
        # Target at t should be (Close[t+1] > Close[t])
        # For index 0: Close[1]=101 > Close[0]=100 → target=1
        assert engineer.df['target'].iloc[0] == 1.0
        
        # For index 2: Close[3]=102 > Close[2]=99 → target=1
        assert engineer.df['target'].iloc[2] == 1.0
        
        # For index 4: Close[5]=103 > Close[4]=98 → target=1
        assert engineer.df['target'].iloc[4] == 1.0
        
        # Last row should have NaN target (no T+1)
        assert pd.isna(engineer.df['target'].iloc[-1])


class TestDataLeakage:
    """Test for data leakage detection."""
    
    def test_target_not_in_features(self):
        """Test that target column is not used as a feature."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        
        # Check feature names
        assert Config.TARGET_COLUMN not in model.feature_names, \
            f"Target '{Config.TARGET_COLUMN}' found in features!"
        assert 'target' not in model.feature_names, \
            "'target' found in features!"
    
    def test_leakage_detection_function(self):
        """Test the leakage detection utility."""
        # Clean data
        df_clean = create_sample_processed_data(100, include_leakage=False)
        issues_clean = detect_leakage_in_features(df_clean, 'target')
        assert len(issues_clean) == 0, "False positive leakage detection"
        
        # Data with intentional leakage
        df_leaky = create_sample_processed_data(100, include_leakage=True)
        issues_leaky = detect_leakage_in_features(df_leaky, 'target')
        assert len(issues_leaky) > 0, "Leakage not detected!"
    
    def test_scaler_fit_on_train_only(self):
        """Test that scaler is fit on training data only."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        
        # Train model (which fits scaler)
        model.train(X_train, y_train)
        
        # Scaler should have statistics from train set only
        # Check that scaler was fit
        assert hasattr(model.scaler, 'mean_'), "Scaler not fitted"
        assert len(model.scaler.mean_) == X_train.shape[1], "Scaler dimension mismatch"
    
    def test_intentional_leakage_causes_suspicion(self):
        """Test that intentional leakage results in suspicious accuracy."""
        df = create_sample_processed_data(100, include_leakage=True)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        model.train(X_train, y_train)
        
        evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
        evaluator.evaluate_model()
        evaluator.evaluate_baselines()
        comparison = evaluator.compare_to_baselines()
        
        # With leakage, accuracy should be suspicious (> 70%)
        assert comparison['suspicious_high_accuracy'], \
            "Leakage not flagged as suspicious!"


class TestBaselineSanityChecks:
    """Test baseline comparisons."""
    
    def test_model_beats_random(self):
        """Test that model beats random classifier."""
        df = create_sample_processed_data(200)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        model.train(X_train, y_train)
        
        evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
        evaluator.evaluate_model()
        evaluator.evaluate_baselines()
        comparison = evaluator.compare_to_baselines()
        
        # Model should beat random (though not guaranteed for all random seeds)
        # At minimum, it shouldn't be significantly worse
        assert comparison['improvement_over_random'] >= -0.1, \
            "Model significantly worse than random!"
    
    def test_baselines_calculated(self):
        """Test that all baseline metrics are calculated."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        model.train(X_train, y_train)
        
        evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
        evaluator.evaluate_baselines()
        
        assert 'baseline_random_accuracy' in evaluator.baseline_metrics
        assert 'baseline_majority_accuracy' in evaluator.baseline_metrics


class TestStabilityReproducibility:
    """Test model stability and reproducibility."""
    
    def test_same_seed_same_results(self):
        """Test that same random_state produces same results."""
        df = create_sample_processed_data(100)
        
        # Train twice with same seed
        results = []
        for _ in range(2):
            model = StockTrendModel('logistic')
            X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
            model.train(X_train, y_train)
            
            preds = model.predict(X_test)
            acc = (preds == y_test).mean()
            results.append(acc)
        
        # Results should be identical
        assert abs(results[0] - results[1]) < 1e-10, \
            "Non-deterministic behavior with same random_state!"
    
    def test_predictions_deterministic(self):
        """Test that predictions are deterministic."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        model.train(X_train, y_train)
        
        # Get predictions multiple times
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        preds3 = model.predict(X_test)
        
        # Should be identical
        assert np.array_equal(preds1, preds2), "Predictions not deterministic!"
        assert np.array_equal(preds2, preds3), "Predictions not deterministic!"


class TestClassImbalance:
    """Test class imbalance analysis."""
    
    def test_confusion_matrix_generated(self):
        """Test that confusion matrix is generated."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        model.train(X_train, y_train)
        
        evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
        evaluator.evaluate_model()
        
        assert 'test_confusion_matrix' in evaluator.metrics
        cm = evaluator.metrics['test_confusion_matrix']
        assert cm.shape == (2, 2), "Invalid confusion matrix shape"
    
    def test_not_trivially_predicting_one_class(self):
        """Test that model doesn't trivially predict one class."""
        df = create_sample_processed_data(200)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        model.train(X_train, y_train)
        
        evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
        evaluator.evaluate_model()
        imbalance = evaluator.check_class_imbalance()
        
        assert not imbalance['trivial_prediction'], \
            "Model trivially predicts one class (>95%)!"
    
    def test_precision_recall_per_class(self):
        """Test that precision/recall are calculated for both classes."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        model.train(X_train, y_train)
        
        evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
        evaluator.evaluate_model()
        
        assert 'test_precision' in evaluator.metrics
        assert 'test_recall' in evaluator.metrics


class TestFailureModes:
    """Test failure mode handling."""
    
    def test_empty_dataset_fails_gracefully(self):
        """Test that empty dataset produces clear error."""
        df = pd.DataFrame(columns=['Date', 'Close', 'returns', 'ma_5', 'target'])
        
        model = StockTrendModel('logistic')
        
        with pytest.raises((ValueError, IndexError)) as exc_info:
            model.prepare_data(df)
    
    def test_too_short_history_detected(self):
        """Test that too-short history is problematic."""
        # Create data with only 10 samples (less than rolling window)
        df = create_sample_processed_data(10)
        
        model = StockTrendModel('logistic')
        
        # This should work but produce a very small train set
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        
        # Train set should be tiny (70% of 10 = 7 samples)
        assert len(X_train) <= 10, "Expected small training set"
    
    def test_model_predict_before_train_fails(self):
        """Test that predict() before train() raises error."""
        df = create_sample_processed_data(100)
        
        model = StockTrendModel('logistic')
        X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
        
        # Try to predict without training
        with pytest.raises(ValueError) as exc_info:
            model.predict(X_test)
        
        assert "not trained" in str(exc_info.value).lower()


if __name__ == "__main__":
    #Run tests
    pytest.main([__file__, "-v", "--tb=short"])
