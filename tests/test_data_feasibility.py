"""
Comprehensive tests for data feasibility validation.
Ensures production-grade data quality checks with hard-fail mechanisms.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_feasibility import (
    DataFeasibilityValidator,
    DataFeasibilityError,
    validate_data_feasibility
)


def create_valid_stock_data(n_samples=200, include_target=True):
    """Create valid stock data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=n_samples, freq='B')  # Business days
    
    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
    close_prices = np.maximum(close_prices, 50)  # Floor at 50
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * np.random.uniform(0.98, 1.02, n_samples),
        'High': close_prices * np.random.uniform(1.00, 1.05, n_samples),
        'Low': close_prices * np.random.uniform(0.95, 1.00, n_samples),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, n_samples)
    })
    
    # Ensure OHLC relationships are valid
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    if include_target:
        # Create balanced binary target
        df['target'] = (df['Close'].pct_change() > 0).astype(int)
        df['target'].iloc[0] = 0  # Fix first NaN
    
    return df


class TestMinimumSamples:
    """Test minimum samples requirement."""
    
    def test_pass_with_sufficient_samples(self):
        """Test that sufficient samples pass."""
        df = create_valid_stock_data(n_samples=150)
        validator = DataFeasibilityValidator(df, "TEST")
        
        assert validator.check_minimum_samples() is True
        assert validator.checks_passed['minimum_samples'] is True
    
    def test_fail_with_insufficient_samples(self):
        """Test that insufficient samples fail."""
        df = create_valid_stock_data(n_samples=50)  # Below MIN_SAMPLES (100)
        validator = DataFeasibilityValidator(df, "TEST")
        
        with pytest.raises(DataFeasibilityError) as exc_info:
            validator.check_minimum_samples()
        
        assert "Minimum samples check FAILED" in str(exc_info.value)
        assert "50" in str(exc_info.value)
    
    def test_exact_threshold(self):
        """Test exactly at minimum threshold."""
        df = create_valid_stock_data(n_samples=100)  # Exactly MIN_SAMPLES
        validator = DataFeasibilityValidator(df, "TEST")
        
        assert validator.check_minimum_samples() is True


class TestCompleteness:
    """Test data completeness checks."""
    
    def test_pass_with_complete_data(self):
        """Test that complete data passes."""
        df = create_valid_stock_data(n_samples=150)
        validator = DataFeasibilityValidator(df, "TEST")
        
        assert validator.check_completeness() is True
        assert validator.check_details['completeness']['completeness_pct'] == 100.0
    
    def test_fail_with_excessive_missing(self):
        """Test that excessive missing data fails."""
        df = create_valid_stock_data(n_samples=150)
        
        # Create 12% missing data total across OHLCV (exceeds 10% threshold)
        # 5 columns, so 12% total = ~18 cells missing
        missing_count = int(len(df) * 5 * 0.12)
        
        # Distribute missing across columns
        for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
            col_missing = missing_count // 5
            indices = np.random.choice(df.index, size=min(col_missing, len(df)), replace=False)
            df.loc[indices, col] = np.nan
        
        validator = DataFeasibilityValidator(df, "TEST")
        
        with pytest.raises(DataFeasibilityError) as exc_info:
            validator.check_completeness()
        
        assert "Completeness check FAILED" in str(exc_info.value)
    
    def test_pass_with_acceptable_missing(self):
        """Test that acceptable missing data passes."""
        df = create_valid_stock_data(n_samples=150)
        
        # Create 5% missing data (within threshold)
        missing_indices = np.random.choice(df.index, size=7, replace=False)
        df.loc[missing_indices, 'Volume'] = np.nan
        
        validator = DataFeasibilityValidator(df, "TEST")
        
        assert validator.check_completeness() is True


class TestTemporalContinuity:
    """Test temporal continuity checks."""
    
    def test_pass_with_no_gaps(self):
        """Test that continuous data passes."""
        df = create_valid_stock_data(n_samples=150)
        validator = DataFeasibilityValidator(df, "TEST")
        
        assert validator.check_temporal_continuity() is True
    
    def test_fail_with_large_gap(self):
        """Test that large gaps fail."""
        df = create_valid_stock_data(n_samples=150)
        
        # Create a 15-day gap (exceeds MAX_GAP_DAYS = 10)
        gap_index = 50
        df.loc[gap_index:, 'Date'] = df.loc[gap_index:, 'Date'] + timedelta(days=15)
        
        validator = DataFeasibilityValidator(df, "TEST")
        
        with pytest.raises(DataFeasibilityError) as exc_info:
            validator.check_temporal_continuity()
        
        assert "Temporal continuity check FAILED" in str(exc_info.value)
        assert "gaps exceeding 10 days" in str(exc_info.value)
    
    def test_pass_with_acceptable_gaps(self):
        """Test that small gaps pass (e.g., weekends)."""
        df = create_valid_stock_data(n_samples=150)
        
        # Small 3-day gap (within threshold)
        gap_index = 75
        df.loc[gap_index:, 'Date'] = df.loc[gap_index:, 'Date'] + timedelta(days=3)
        
        validator = DataFeasibilityValidator(df, "TEST")
        
        assert validator.check_temporal_continuity() is True


class TestOutliers:
    """Test outlier detection."""
    
    def test_pass_with_normal_returns(self):
        """Test that normal data passes."""
        df = create_valid_stock_data(n_samples=150)
        validator = DataFeasibilityValidator(df, "TEST")
        
        assert validator.check_outliers() is True
    
    def test_fail_with_excessive_outliers(self):
        """Test that excessive outliers fail."""
        df = create_valid_stock_data(n_samples=150)
        
        # Create extreme outliers (>1% of data = need at least 2 out of 150)
        # Create 3 outliers to be safe (2% > 1% threshold)
        outlier_indices = [30, 60, 90]
        for idx in outlier_indices:
            df.loc[idx, 'Close'] = df.loc[idx, 'Close'] * 10  # 10x price spike
        
        validator = DataFeasibilityValidator(df, "TEST")
        
        with pytest.raises(DataFeasibilityError) as exc_info:
            validator.check_outliers()
        
        assert "Outlier check FAILED" in str(exc_info.value)
    
    def test_pass_with_few_outliers(self):
        """Test that acceptable number of outliers passes."""
        df = create_valid_stock_data(n_samples=200)
        
        # Create only 1 outlier (<1%)
        df.loc[100, 'Close'] = df.loc[100, 'Close'] * 3
        
        validator = DataFeasibilityValidator(df, "TEST")
        
        assert validator.check_outliers() is True


class TestLabelBalance:
    """Test label balance checks."""
    
    def test_pass_with_balanced_labels(self):
        """Test that balanced labels pass."""
        df = create_valid_stock_data(n_samples=150, include_target=True)
        validator = DataFeasibilityValidator(df, "TEST", target_col='target')
        
        assert validator.check_label_balance() is True
    
    def test_fail_with_severe_imbalance(self):
        """Test that severe imbalance fails."""
        df = create_valid_stock_data(n_samples=150)
        
        # Create 95% class 0, 5% class 1 (below MIN_LABEL_BALANCE = 20%)
        df['target'] = 0
        minority_count = int(len(df) * 0.05)
        df.loc[df.index[:minority_count], 'target'] = 1
        
        validator = DataFeasibilityValidator(df, "TEST", target_col='target')
        
        with pytest.raises(DataFeasibilityError) as exc_info:
            validator.check_label_balance()
        
        assert "Label balance check FAILED" in str(exc_info.value)
        assert "below minimum" in str(exc_info.value)
    
    def test_pass_with_acceptable_imbalance(self):
        """Test that acceptable imbalance passes."""
        df = create_valid_stock_data(n_samples=150)
        
        # Create 70% class 0, 30% class 1 (within range)
        df['target'] = 0
        minority_count = int(len(df) * 0.30)
        df.loc[df.index[:minority_count], 'target'] = 1
        
        validator = DataFeasibilityValidator(df, "TEST", target_col='target')
        
        assert validator.check_label_balance() is True
    
    def test_skip_when_no_target(self):
        """Test that check is skipped when no target column."""
        df = create_valid_stock_data(n_samples=150, include_target=False)
        validator = DataFeasibilityValidator(df, "TEST", target_col=None)
        
        assert validator.check_label_balance() is True


class TestLookaheadBias:
    """Test look-ahead bias detection."""
    
    def test_pass_with_clean_features(self):
        """Test that clean features pass."""
        df = create_valid_stock_data(n_samples=150, include_target=True)
        validator = DataFeasibilityValidator(df, "TEST", target_col='target')
        
        assert validator.check_lookahead_bias() is True
    
    def test_fail_with_suspicious_columns(self):
        """Test that suspicious column names fail."""
        df = create_valid_stock_data(n_samples=150, include_target=True)
        
        # Add suspicious column
        df['future_price'] = df['Close'].shift(-1)  # Look-ahead!
        
        validator = DataFeasibilityValidator(df, "TEST", target_col='target')
        
        with pytest.raises(DataFeasibilityError) as exc_info:
            validator.check_lookahead_bias()
        
        assert "Look-ahead bias check FAILED" in str(exc_info.value)
        assert "future_price" in str(exc_info.value)
    
    def test_skip_when_no_target(self):
        """Test that check is skipped when no target."""
        df = create_valid_stock_data(n_samples=150)
        validator = DataFeasibilityValidator(df, "TEST", target_col=None)
        
        assert validator.check_lookahead_bias() is True


class TestValidateAll:
    """Test comprehensive validation."""
    
    def test_pass_with_valid_data(self):
        """Test that valid data passes all checks."""
        df = create_valid_stock_data(n_samples=200, include_target=True)
        validator = DataFeasibilityValidator(df, "TEST", target_col='target')
        
        summary = validator.validate_all()
        
        assert summary['verdict'] == 'PASS'
        assert summary['production_ready'] is True
        assert all(summary['checks_passed'].values())
    
    def test_fail_on_any_check_failure(self):
        """Test that validation fails if ANY check fails."""
        df = create_valid_stock_data(n_samples=50)  # Too small
        validator = DataFeasibilityValidator(df, "TEST")
        
        with pytest.raises(DataFeasibilityError):
            validator.validate_all()
    
    def test_convenience_function(self):
        """Test convenience function."""
        df = create_valid_stock_data(n_samples=200, include_target=True)
        
        summary = validate_data_feasibility(df, "TEST", target_col='target')
        
        assert summary['verdict'] == 'PASS'
        assert summary['ticker'] == 'TEST'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
