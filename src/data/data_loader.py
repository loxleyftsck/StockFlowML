"""
Enhanced data loader with CSV fallback mechanism.
Handles data acquisition from Yahoo Finance with explicit fallback to CSV snapshots.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging
import time

from src.utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """Load stock data from Yahoo Finance with CSV fallback."""
    
    def __init__(self, ticker: str = Config.DEFAULT_TICKER):
        """
        Initialize data loader.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'BBCA.JK' for BCA Indonesia)
        """
        self.ticker = ticker
        self.data: Optional[pd.DataFrame] = None
        self.data_source: Optional[str] = None  # Track which source was used
    
    def _validate_dataframe(self, df: pd.DataFrame, source: str) -> bool:
        """
        Validate that DataFrame contains valid stock data.
        
        Args:
            df: DataFrame to validate
            source: Data source name (for logging)
            
        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            logger.warning(f"{source}: DataFrame is empty")
            return False
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"{source}: Missing required columns: {missing_cols}")
            return False
        
        # Check for all-NaN data
        if df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().all().all():
            logger.warning(f"{source}: All OHLCV values are NaN")
            return False
        
        logger.info(f"{source}: DataFrame valid ({len(df)} rows)")
        return True
    
    def _download_from_yahoo(
        self,
        years: int,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """
        Attempt to download data from Yahoo Finance.
        
        Returns:
            DataFrame if successful, None if failed
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start = datetime.now() - timedelta(days=years * 365)
            start_date = start.strftime('%Y-%m-%d')
        
        logger.info(f"Attempting Yahoo Finance download: {self.ticker} ({start_date} to {end_date})")
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                ticker_obj = yf.Ticker(self.ticker)
                df = ticker_obj.history(start=start_date, end=end_date)
                
                if df.empty:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Yahoo Finance attempt {attempt + 1}/{max_retries}: empty response for {self.ticker}"
                        )
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(
                            f"Yahoo Finance FAILED after {max_retries} attempts: {self.ticker} returned empty DataFrame"
                        )
                        return None
                
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Select relevant columns
                columns_needed = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                df = df[columns_needed]
                
                if self._validate_dataframe(df, "Yahoo Finance"):
                    logger.info(f"✓ Yahoo Finance SUCCESS: Downloaded {len(df)} rows for {self.ticker}")
                    return df
                else:
                    return None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Yahoo Finance attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Yahoo Finance FAILED: {e}")
                    return None
        
        return None
    
    def _load_from_fallback(self) -> Optional[pd.DataFrame]:
        """
        Load data from CSV fallback snapshot.
        
        Returns:
            DataFrame if successful, None if failed
        """
        fallback_path = Config.get_fallback_data_path(self.ticker)
        
        if not fallback_path.exists():
            logger.error(f"Fallback CSV NOT FOUND: {fallback_path}")
            logger.error(
                f"To create fallback snapshot, download data manually and save to:\n  {fallback_path}"
            )
            return None
        
        logger.warning(f"⚠️  Loading FALLBACK data from: {fallback_path}")
        
        try:
            df = pd.read_csv(fallback_path, parse_dates=['Date'])
            
            if self._validate_dataframe(df, "Fallback CSV"):
                logger.info(f"✓ Fallback CSV SUCCESS: Loaded {len(df)} rows")
                logger.warning("⚠️  Using SNAPSHOT data - not real-time market data!")
                return df
            else:
                logger.error("Fallback CSV validation FAILED")
                return None
                
        except Exception as e:
            logger.error(f"Fallback CSV load FAILED: {e}")
            return None
    
    def download_data(
        self, 
        years: int = Config.LOOKBACK_YEARS,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_fallback: bool = Config.USE_FALLBACK
    ) -> pd.DataFrame:
        """
        Download historical stock data with fallback mechanism.
        
        Data Source Priority:
        1. Yahoo Finance (primary)
        2. CSV snapshot (fallback, if use_fallback=True)
        3. RuntimeError (if both fail)
        
        Args:
            years: Number of years of historical data
            start_date: Start date (YYYY-MM-DD format), overrides years
            end_date: End date (YYYY-MM-DD format), defaults to today
            use_fallback: Enable CSV fallback if Yahoo Finance fails
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            RuntimeError: If all data sources fail
        """
        logger.info("="*70)
        logger.info(f"DATA ACQUISITION: {self.ticker}")
        logger.info("="*70)
        
        # Try Yahoo Finance first
        df = self._download_from_yahoo(years, start_date, end_date)
        
        if df is not None:
            self.data = df
            self.data_source = "Yahoo Finance"
            logger.info(f"✓ PRIMARY source used: Yahoo Finance")
            return df
        
        # Yahoo Finance failed, try fallback
        logger.warning("-"*70)
        logger.warning("PRIMARY SOURCE FAILED: Yahoo Finance unavailable")
        logger.warning("-"*70)
        
        if not use_fallback:
            raise RuntimeError(
                f"Yahoo Finance failed for {self.ticker} and fallback is disabled. "
                f"Enable fallback with use_fallback=True or set Config.USE_FALLBACK=True"
            )
        
        logger.warning("Attempting FALLBACK: CSV snapshot")
        df = self._load_from_fallback()
        
        if df is not None:
            self.data = df
            self.data_source = "CSV Fallback"
            logger.warning("="*70)
            logger.warning(f"⚠️  FALLBACK source used: CSV Snapshot")
            logger.warning(f"⚠️  Data may be STALE - verify snapshot date!")
            logger.warning("="*70)
            return df
        
        # Both sources failed
        logger.error("="*70)
        logger.error("💥 ALL DATA SOURCES FAILED")
        logger.error("="*70)
        raise RuntimeError(
            f"Failed to download data for {self.ticker} from all sources:\\n"
            f"  1. Yahoo Finance: FAILED (empty response or error)\\n"
            f"  2. CSV Fallback: FAILED (file not found or invalid)\\n"
            f"\\nTo resolve:\\n"
        f"  - Create fallback CSV at: {Config.get_fallback_data_path(self.ticker)}"
        )
    
    def clean_data(self, df: Optional[pd.DataFrame] = None, validate: bool = True) -> pd.DataFrame:
        """
        Clean the downloaded data with optional validation.
        
        Args:
            df: DataFrame to clean, uses self.data if None
            validate: If True, run full data contract validation
            
        Returns:
            Cleaned and validated DataFrame
            
        Raises:
            DataValidationError: If validation fails
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data to clean. Run download_data() first.")
        
        logger.info("Cleaning data...")
        
        # Remove any duplicates
        df = df.drop_duplicates(subset=['Date'], keep='last')
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Handle missing values (forward fill, then backward fill)
        df = df.ffill().bfill()
        
        # Remove rows with zero volume (likely holidays)
        df = df[df['Volume'] > 0]
        
        logger.info(f"Data cleaned. {len(df)} rows remaining")
        
        # Run validation if requested
        if validate:
            from src.data.data_validation import validate_dataframe
            logger.info("Running data contract validation...")
            df, validation_stats = validate_dataframe(df, self.ticker)
            self.validation_stats = validation_stats
            logger.info("✓ Data validation passed")
        
        self.data = df
        return df
    
    def save_data(self, output_path: Optional[Path] = None):
        """
        Save data to CSV.
        
        Args:
            output_path: Path to save CSV, defaults to Config path
        """
        if self.data is None:
            raise ValueError("No data to save. Run download_data() first.")
        
        if output_path is None:
            output_path = Config.get_raw_data_path(self.ticker)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(output_path, index=False)
        
        logger.info(f"Data saved to {output_path}")
        if self.data_source:
            logger.info(f"Data source: {self.data_source}")
    
    def load_from_file(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load data from existing CSV file.
        
        Args:
            file_path: Path to CSV file, defaults to Config path
            
        Returns:
            Loaded DataFrame
        """
        if file_path is None:
            file_path = Config.get_raw_data_path(self.ticker)
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        self.data = df
        self.data_source = "Local CSV"
        return df


def main():
    """Main function for testing data loader."""
    loader = StockDataLoader(ticker=Config.DEFAULT_TICKER)
    
    # Download and clean data (with fallback enabled)
    df = loader.download_data()
    df = loader.clean_data()
    
    # Save to file
    loader.save_data()
    
    print(f"\nData shape: {df.shape}")
    print(f"Data source: {loader.data_source}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData info:")
    print(df.info())
    print(f"\nBasic statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
