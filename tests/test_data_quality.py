"""
Comprehensive data quality tests for StockFlowML.
Tests data contract enforcement and validation logic.
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

from src.data.data_validation import DataValidator, DataValidationError, validate_dataframe
from src.utils.config import Config


def create_valid_stock_data(n_rows=30) -> pd.DataFrame:
    """Create valid stock data for testing."""
    dates = pd.date_range('2021-01-04', periods=n_rows, freq='D')
    # Filter to weekdays
    dates = [d for d in dates if d.weekday() < 5][:n_rows]
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Ensure valid High/Low
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(1, 5, len(dates))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(1, 5, len(dates))
    
    return df


class TestDataValidation:
    """Test suite for data validation module."""
    
    def test_valid_data_passes(self):
        """Test that valid data passes all checks."""
        df = create_valid_stock_data()
        validator = DataValidator(df, "TEST")
        
        df_clean, stats = validator.validate_all()
        
        assert len(df_clean) > 0
        assert stats['rows_removed'] == 0
        assert stats['ticker'] == 'TEST'
    
    def test_missing_column_fails(self):
        """Test that missing required columns fails schema validation."""
        df = create_valid_stock_data()
        df = df.drop('Close', axis=1)
        
        validator = DataValidator(df, "TEST")
        
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate_schema()
        
        assert "Missing required columns" in str(exc_info.value)
        assert "Close" in str(exc_info.value)
    
    def test_duplicate_dates_fails(self):
        """Test that duplicate timestamps fail validation."""
        df = create_valid_stock_data()
        # Duplicate first row
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()  # Should pass
        
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate_time_properties()
        
        assert "duplicate timestamps" in str(exc_info.value)
    
    def test_unsorted_dates_fails(self):
        """Test that unsorted dates fail validation."""
        df = create_valid_stock_data()
        # Reverse the dates
        df['Date'] = df['Date'][::-1].values
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate_time_properties()
        
        assert "not sorted" in str(exc_info.value)
    
    def test_negative_prices_removed(self):
        """Test that negative prices are removed."""
        df = create_valid_stock_data()
        # Set some negative prices
        df.loc[0, 'Open'] = -100
        df.loc[1, 'Close'] = -50
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        df_clean = validator.validate_financial_integrity()
        
        assert len(df_clean) < len(df)
        assert validator.rows_removed == 2
        assert 'negative_or_zero_prices' in validator.invalid_rows
    
    def test_zero_prices_removed(self):
        """Test that zero prices are removed."""
        df = create_valid_stock_data()
        df.loc[0, 'High'] = 0
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        df_clean = validator.validate_financial_integrity()
        
        assert len(df_clean) < len(df)
        assert validator.rows_removed >= 1
    
    def test_high_less_than_low_removed(self):
        """Test that High < Low violations are removed."""
        df = create_valid_stock_data()
        # Swap High and Low for some rows
        df.loc[0, ['High', 'Low']] = [50, 150]  # High=50, Low=150 (invalid)
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        df_clean = validator.validate_financial_integrity()
        
        assert len(df_clean) < len(df)
        assert 'high_less_than_low' in validator.invalid_rows
    
    def test_high_less_than_open_removed(self):
        """Test that High < Open violations are removed."""
        df = create_valid_stock_data()
        df.loc[0, 'Open'] = 200
        df.loc[0, 'High'] = 100  # High less than Open
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        df_clean = validator.validate_financial_integrity()
        
        assert len(df_clean) < len(df)
    
    def test_high_less_than_close_removed(self):
        """Test that High < Close violations are removed."""
        df = create_valid_stock_data()
        df.loc[0, 'Close'] = 200
        df.loc[0, 'High'] = 100  # High less than Close
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        df_clean = validator.validate_financial_integrity()
        
        assert len(df_clean) < len(df)
    
    def test_low_greater_than_open_removed(self):
        """Test that Low > Open violations are removed."""
        df = create_valid_stock_data()
        df.loc[0, 'Open'] = 100
        df.loc[0, 'Low'] = 200  # Low greater than Open
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        df_clean = validator.validate_financial_integrity()
        
        assert len(df_clean) < len(df)
    
    def test_low_greater_than_close_removed(self):
        """Test that Low > Close violations are removed."""
        df = create_valid_stock_data()
        df.loc[0, 'Close'] = 100
        df.loc[0, 'Low'] = 200  # Low greater than Close
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        df_clean = validator.validate_financial_integrity()
        
        assert len(df_clean) < len(df)
    
    def test_too_many_invalid_rows_fails(self):
        """Test that too many invalid rows (>10%) fails validation."""
        df = create_valid_stock_data(100)
        # Make 15% of rows invalid
        df.loc[:14, 'Open'] = -100
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate_financial_integrity()
        
        assert "Removed" in str(exc_info.value)
        assert "10%" in str(exc_info.value)
    
    def test_missing_ohlcv_fails(self):
        """Test that missing OHLCV values fail validation."""
        df = create_valid_stock_data()
        df.loc[0, 'Close'] = np.nan
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        # Don't call validate_financial_integrity as it removes invalid rows
        
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate_missing_data()
        
        assert "missing values" in str(exc_info.value)
    
    def test_negative_volume_removed(self):
        """Test that negative volume is removed."""
        df = create_valid_stock_data()
        df.loc[0, 'Volume'] = -1000
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        validator.validate_time_properties()
        
        df_clean = validator.validate_financial_integrity()
        
        assert len(df_clean) < len(df)
        assert 'negative_volume' in validator.invalid_rows
    
    def test_date_type_conversion(self):
        """Test that Date column is converted to datetime."""
        df = create_valid_stock_data()
        df['Date'] = df['Date'].astype(str)  # Convert to string
        
        validator = DataValidator(df, "TEST")
        validator.validate_schema()
        
        assert pd.api.types.is_datetime64_any_dtype(validator.df['Date'])
    
    def test_summary_statistics_generated(self):
        """Test that summary statistics are correctly generated."""
        df = create_valid_stock_data()
        validator = DataValidator(df, "TEST")
        
        df_clean, stats = validator.validate_all()
        
        assert 'ticker' in stats
        assert 'total_rows' in stats
        assert 'date_range' in stats
        assert 'price_stats' in stats
        assert 'volume_stats' in stats
        assert stats['ticker'] == 'TEST'
    
    def test_validation_log_populated(self):
        """Test that validation log tracks all checks."""
        df = create_valid_stock_data()
        validator = DataValidator(df, "TEST")
        
        validator.validate_all()
        
        assert len(validator.validation_log) > 0
        assert any('Schema valid' in log for log in validator.validation_log)
    
    def test_empty_dataframe_fails(self):
        """Test that empty DataFrame fails validation."""
        df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Empty DataFrame should fail when checking for missing columns
        # because even though columns exist, we need data
        with pytest.raises((DataValidationError, ValueError)):
            validate_dataframe(df, "TEST")
    
    def test_deterministic_behavior(self):
        """Test that validation is deterministic."""
        df = create_valid_stock_data(seed=42)
        
        df_clean1, stats1 = validate_dataframe(df.copy(), "TEST")
        df_clean2, stats2 = validate_dataframe(df.copy(), "TEST")
        
        assert len(df_clean1) == len(df_clean2)
        assert stats1['total_rows'] == stats2['total_rows']


def create_valid_stock_data(n_rows=30, seed=None) -> pd.DataFrame:
    """Create valid stock data for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range('2021-01-04', periods=n_rows * 2, freq='D')
    # Filter to weekdays
    dates = [d for d in dates if d.weekday() < 5][:n_rows]
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Ensure valid High/Low
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(1, 5, len(dates))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(1, 5, len(dates))
    
    return df


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
