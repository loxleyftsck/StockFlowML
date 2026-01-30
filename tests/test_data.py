"""
Basic tests for data loading functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.data_loader import StockDataLoader
from src.utils.config import Config


def test_data_loader_initialization():
    """Test that data loader initializes correctly."""
    loader = StockDataLoader(ticker="TEST.JK")
    assert loader.ticker == "TEST.JK"
    assert loader.data is None


def test_download_data_structure():
    """Test that downloaded data has correct structure."""
    loader = StockDataLoader(ticker=Config.DEFAULT_TICKER)
    
    # Download limited data for faster testing
    df = loader.download_data(years=1)
    
    # Check columns
    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert all(col in df.columns for col in expected_columns)
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])
    assert pd.api.types.is_numeric_dtype(df['Close'])
    
    # Check no empty dataframe
    assert len(df) > 0


def test_clean_data():
    """Test data cleaning functionality."""
    loader = StockDataLoader(ticker=Config.DEFAULT_TICKER)
    df = loader.download_data(years=1)
    df_cleaned = loader.clean_data(df)
    
    # Should have no NaN values
    assert df_cleaned.isna().sum().sum() == 0
    
    # Should be sorted by date
    assert df_cleaned['Date'].is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
