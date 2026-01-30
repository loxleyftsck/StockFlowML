"""
Data loader module for downloading stock data from Yahoo Finance.
Handles data acquisition and initial cleaning.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging

from src.utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """Load stock data from Yahoo Finance."""
    
    def __init__(self, ticker: str = Config.DEFAULT_TICKER):
        """
        Initialize data loader.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'BBCA.JK' for BCA Indonesia)
        """
        self.ticker = ticker
        self.data: Optional[pd.DataFrame] = None
    
    def download_data(
        self, 
        years: int = Config.LOOKBACK_YEARS,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download historical stock data.
        
        Args:
            years: Number of years of historical data
            start_date: Start date (YYYY-MM-DD format), overrides years
            end_date: End date (YYYY-MM-DD format), defaults to today
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start = datetime.now() - timedelta(days=years * 365)
            start_date = start.strftime('%Y-%m-%d')
        
        logger.info(f"Downloading {self.ticker} from {start_date} to {end_date}")
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            df = ticker_obj.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Select relevant columns
            columns_needed = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[columns_needed]
            
            logger.info(f"Downloaded {len(df)} rows of data")
            
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to download data for {self.ticker}: {e}")
            raise
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the downloaded data.
        
        Args:
            df: DataFrame to clean, uses self.data if None
            
        Returns:
            Cleaned DataFrame
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
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with zero volume (likely holidays)
        df = df[df['Volume'] > 0]
        
        logger.info(f"Data cleaned. {len(df)} rows remaining")
        
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
        return df


def main():
    """Main function for testing data loader."""
    loader = StockDataLoader(ticker=Config.DEFAULT_TICKER)
    
    # Download and clean data
    df = loader.download_data()
    df = loader.clean_data()
    
    # Save to file
    loader.save_data()
    
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData info:")
    print(df.info())
    print(f"\nBasic statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
