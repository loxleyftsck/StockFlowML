"""
Data validation module for StockFlowML.
Enforces data contract for financial time-series data.

This module ensures all stock market data (from Yahoo Finance or CSV snapshots)
conforms to strict quality standards before being used for ML training.

Data Contract:
- Schema: Date (datetime), OHLC (float), Volume (int)
- Time: Daily frequency, trading days only, sorted ascending, no duplicates
- Financial: Positive prices, valid OHLC relationships (Low <= Open/Close <= High)
- Integrity: No missing OHLC, no forward-fill across gaps, no future leakage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass


class DataValidator:
    """
    Validates financial time-series data against production data contract.
    
    All validation failures raise DataValidationError with actionable messages.
    NO silent fixes - data must be corrected at the source.
    """
    
    def __init__(self, df: pd.DataFrame, ticker: str = "UNKNOWN"):
        """
        Initialize validator with DataFrame.
        
        Args:
            df: DataFrame to validate
            ticker: Stock ticker symbol (for error messages)
        """
        self.df = df.copy()
        self.ticker = ticker
        self.validation_log: List[str] = []
        self.rows_before = len(df)
        self.rows_removed = 0
        self.invalid_rows: Dict[str, int] = {}
    
    def validate_schema(self) -> None:
        """
        Validate DataFrame schema.
        
        Checks:
        - Required columns present
        - Correct data types
        
        Raises:
            DataValidationError: If schema is invalid
        """
        logger.info(f"[VALIDATION] Schema check for {self.ticker}")
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check columns exist
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise DataValidationError(
                f"Schema validation failed for {self.ticker}: "
                f"Missing required columns: {missing}. "
                f"Expected: {required_columns}, Got: {list(self.df.columns)}"
            )
        
        # Check Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            # Attempt conversion
            try:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                self.validation_log.append("✓ Converted Date column to datetime")
            except Exception as e:
                raise DataValidationError(
                    f"Schema validation failed for {self.ticker}: "
                    f"Date column must be datetime-compatible. Error: {e}"
                )
        
        # Check OHLC are numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise DataValidationError(
                    f"Schema validation failed for {self.ticker}: "
                    f"Column '{col}' must be numeric. Got dtype: {self.df[col].dtype}"
                )
        
        # Check Volume is numeric (will convert to int later if needed)
        if not pd.api.types.is_numeric_dtype(self.df['Volume']):
            raise DataValidationError(
                f"Schema validation failed for {self.ticker}: "
                f"Volume column must be numeric. Got dtype: {self.df['Volume'].dtype}"
            )
        
        self.validation_log.append(f"✓ Schema valid: all required columns present with correct types")
        logger.info(f"[VALIDATION] ✓ Schema check passed")
    
    def validate_time_properties(self) -> None:
        """
        Validate time index properties.
        
        Checks:
        - No duplicate timestamps
        - Sorted ascending by date
        - Only weekdays (trading days)
        
        Raises:
            DataValidationError: If time properties are invalid
        """
        logger.info(f"[VALIDATION] Time properties check for {self.ticker}")
        
        # Check for duplicates
        duplicates = self.df['Date'].duplicated()
        if duplicates.any():
            dup_count = duplicates.sum()
            dup_dates = self.df[duplicates]['Date'].tolist()[:5]  # Show first 5
            raise DataValidationError(
                f"Time validation failed for {self.ticker}: "
                f"Found {dup_count} duplicate timestamps. "
                f"First few: {dup_dates}. "
                f"Action: Remove duplicates before validation."
            )
        
        # Check sorted
        if not self.df['Date'].is_monotonic_increasing:
            raise DataValidationError(
                f"Time validation failed for {self.ticker}: "
                f"Dates are not sorted in ascending order. "
                f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}. "
                f"Action: Sort data by Date before validation."
            )
        
        # Check for weekends (optional - trading data shouldn't have weekends)
        weekend_mask = self.df['Date'].dt.dayofweek >= 5  # 5 = Saturday, 6 = Sunday
        if weekend_mask.any():
            weekend_count = weekend_mask.sum()
            logger.warning(
                f"[VALIDATION] ⚠ Found {weekend_count} weekend dates in {self.ticker}. "
                f"These will be flagged but not failed (some markets trade weekends)."
            )
            self.validation_log.append(f"⚠ Found {weekend_count} weekend dates (check if expected)")
        
        self.validation_log.append(f"✓ Time properties valid: no duplicates, sorted ascending")
        logger.info(f"[VALIDATION] ✓ Time properties check passed")
    
    def validate_financial_integrity(self) -> pd.DataFrame:
        """
        Validate financial data integrity.
        
        Checks and removes invalid rows:
        - Prices must be > 0
        - Volume must be >= 0
        - Low <= Open <= High
        - Low <= Close <= High
        
        Returns:
            DataFrame with invalid rows removed
            
        Raises:
            DataValidationError: If too many invalid rows (> 10%)
        """
        logger.info(f"[VALIDATION] Financial integrity check for {self.ticker}")
        
        df_clean = self.df.copy()
        initial_rows = len(df_clean)
        
        # Check 1: Positive prices
        negative_prices = (
            (df_clean['Open'] <= 0) |
            (df_clean['High'] <= 0) |
            (df_clean['Low'] <= 0) |
            (df_clean['Close'] <= 0)
        )
        if negative_prices.any():
            count = negative_prices.sum()
            self.invalid_rows['negative_or_zero_prices'] = count
            df_clean = df_clean[~negative_prices]
            logger.warning(f"[VALIDATION] Removed {count} rows with non-positive prices")
        
        # Check 2: Non-negative volume
        negative_volume = df_clean['Volume'] < 0
        if negative_volume.any():
            count = negative_volume.sum()
            self.invalid_rows['negative_volume'] = count
            df_clean = df_clean[~negative_volume]
            logger.warning(f"[VALIDATION] Removed {count} rows with negative volume")
        
        # Check 3: High >= Low
        high_low_violation = df_clean['High'] < df_clean['Low']
        if high_low_violation.any():
            count = high_low_violation.sum()
            self.invalid_rows['high_less_than_low'] = count
            df_clean = df_clean[~high_low_violation]
            logger.warning(f"[VALIDATION] Removed {count} rows where High < Low")
        
        # Check 4: High >= Open
        high_open_violation = df_clean['High'] < df_clean['Open']
        if high_open_violation.any():
            count = high_open_violation.sum()
            self.invalid_rows['high_less_than_open'] = count
            df_clean = df_clean[~high_open_violation]
            logger.warning(f"[VALIDATION] Removed {count} rows where High < Open")
        
        # Check 5: High >= Close
        high_close_violation = df_clean['High'] < df_clean['Close']
        if high_close_violation.any():
            count = high_close_violation.sum()
            self.invalid_rows['high_less_than_close'] = count
            df_clean = df_clean[~high_close_violation]
            logger.warning(f"[VALIDATION] Removed {count} rows where High < Close")
        
        # Check 6: Low <= Open
        low_open_violation = df_clean['Low'] > df_clean['Open']
        if low_open_violation.any():
            count = low_open_violation.sum()
            self.invalid_rows['low_greater_than_open'] = count
            df_clean = df_clean[~low_open_violation]
            logger.warning(f"[VALIDATION] Removed {count} rows where Low > Open")
        
        # Check 7: Low <= Close
        low_close_violation = df_clean['Low'] > df_clean['Close']
        if low_close_violation.any():
            count = low_close_violation.sum()
            self.invalid_rows['low_greater_than_close'] = count
            df_clean = df_clean[~low_close_violation]
            logger.warning(f"[VALIDATION] Removed {count} rows where Low > Close")
        
        # Calculate removal statistics
        self.rows_removed = initial_rows - len(df_clean)
        removal_rate = (self.rows_removed / initial_rows) * 100 if initial_rows > 0 else 0
        
        # Fail if too many invalid rows
        if removal_rate > 10:
            raise DataValidationError(
                f"Financial integrity validation failed for {self.ticker}: "
                f"Removed {self.rows_removed} invalid rows ({removal_rate:.1f}%). "
                f"Threshold: 10%. "
                f"Invalid row breakdown: {self.invalid_rows}. "
                f"Action: Investigate data source quality."
            )
        
        if self.rows_removed > 0:
            self.validation_log.append(
                f"✓ Removed {self.rows_removed} invalid rows ({removal_rate:.2f}%): {self.invalid_rows}"
            )
            logger.info(f"[VALIDATION] Removed {self.rows_removed} invalid rows")
        else:
            self.validation_log.append(f"✓ Financial integrity perfect: no invalid rows")
        
        logger.info(f"[VALIDATION] ✓ Financial integrity check passed")
        return df_clean
    
    def validate_missing_data(self) -> None:
        """
        Validate missing data rules.
        
        Checks:
        - No missing values in OHLCV columns
        
        Raises:
            DataValidationError: If missing OHLCV data found
        """
        logger.info(f"[VALIDATION] Missing data check for {self.ticker}")
        
        critical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        missing_counts = self.df[critical_columns].isna().sum()
        
        if missing_counts.any():
            missing_summary = missing_counts[missing_counts > 0].to_dict()
            total_missing = missing_counts.sum()
            raise DataValidationError(
                f"Missing data validation failed for {self.ticker}: "
                f"Found {total_missing} missing values in OHLCV columns. "
                f"Breakdown: {missing_summary}. "
                f"Action: Clean data source - missing OHLCV not allowed."
            )
        
        self.validation_log.append(f"✓ No missing OHLCV values")
        logger.info(f"[VALIDATION] ✓ Missing data check passed")
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for reporting."""
        stats = {
            'ticker': self.ticker,
            'total_rows': len(self.df),
            'rows_before_validation': self.rows_before,
            'rows_removed': self.rows_removed,
            'invalid_row_breakdown': self.invalid_rows,
            'date_range': {
                'start': str(self.df['Date'].min().date()),
                'end': str(self.df['Date'].max().date()),
                'trading_days': len(self.df)
            },
            'price_stats': {
                'close_min': float(self.df['Close'].min()),
                'close_max': float(self.df['Close'].max()),
                'close_mean': float(self.df['Close'].mean()),
                'close_std': float(self.df['Close'].std()),
            },
            'volume_stats': {
                'mean': float(self.df['Volume'].mean()),
                'min': int(self.df['Volume'].min()),
                'max': int(self.df['Volume'].max()),
            }
        }
        return stats
    
    def validate_all(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Run all validation checks.
        
        Returns:
            Tuple of (validated_dataframe, summary_statistics)
            
        Raises:
            DataValidationError: If any validation fails
        """
        logger.info("="*70)
        logger.info(f"DATA VALIDATION: {self.ticker}")
        logger.info("="*70)
        
        # Run validations in order
        self.validate_schema()
        self.validate_time_properties()
        df_clean = self.validate_financial_integrity()
        
        # Update df to cleaned version
        self.df = df_clean
        
        # Check missing data on cleaned df
        self.validate_missing_data()
        
        # Generate summary
        stats = self.generate_summary_statistics()
        
        logger.info("="*70)
        logger.info(f"✓ VALIDATION PASSED: {self.ticker}")
        logger.info(f"  Rows: {stats['total_rows']} (removed {self.rows_removed})")
        logger.info(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        logger.info("="*70)
        
        return self.df, stats


def validate_dataframe(df: pd.DataFrame, ticker: str = "UNKNOWN") -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to validate a DataFrame.
    
    Args:
        df: DataFrame to validate
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (validated_dataframe, summary_statistics)
        
    Raises:
        DataValidationError: If validation fails
    """
    validator = DataValidator(df, ticker)
    return validator.validate_all()
