"""
Feature engineering module for creating predictive features.
Implements rolling window features for stock trend prediction.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

from src.utils.config import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for stock trend prediction."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer.
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
    
    def create_returns(self) -> pd.DataFrame:
        """
        Create daily returns feature.
        
        Returns = (Close[t] - Close[t-1]) / Close[t-1]
        """
        logger.info("Creating returns feature...")
        self.df['returns'] = self.df['Close'].pct_change()
        return self.df
    
    def create_rolling_means(self, windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling mean features for different window sizes.
        
        Args:
            windows: List of window sizes (default from Config)
            
        Returns:
            DataFrame with added rolling mean columns
        """
        if windows is None:
            windows = Config.ROLLING_WINDOWS
        
        logger.info(f"Creating rolling means: {windows}")
        
        for window in windows:
            col_name = f'ma_{window}'
            self.df[col_name] = self.df['Close'].rolling(window=window).mean()
        
        return self.df
    
    def create_rolling_volatility(self, windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling volatility (standard deviation of returns).
        
        Args:
            windows: List of window sizes (default from Config)
            
        Returns:
            DataFrame with added volatility columns
        """
        if windows is None:
            windows = Config.ROLLING_WINDOWS
        
        logger.info(f"Creating rolling volatility: {windows}")
        
        # Ensure returns exist
        if 'returns' not in self.df.columns:
            self.create_returns()
        
        for window in windows:
            col_name = f'volatility_{window}'
            self.df[col_name] = self.df['returns'].rolling(window=window).std()
        
        return self.df
    
    def create_target(self) -> pd.DataFrame:
        """
        Create binary target variable.
        
        Target = 1 if Close[t+1] > Close[t], else 0
        
        This represents whether the stock price will go UP tomorrow.
        """
        logger.info("Creating target variable...")
        
        # Shift Close price backward by 1 to get tomorrow's price
        self.df['close_next'] = self.df['Close'].shift(-1)
        
        # Binary target: 1 if price goes up, 0 if down
        self.df[Config.TARGET_COLUMN] = (
            self.df['close_next'] > self.df['Close']
        ).astype(int)
        
        # Drop the helper column
        self.df = self.df.drop('close_next', axis=1)
        
        return self.df
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create all features in one go.
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Creating all features...")
        
        self.create_returns()
        self.create_rolling_means()
        self.create_rolling_volatility()
        self.create_target()
        
        # Drop rows with NaN values (from rolling windows)
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        dropped_rows = initial_rows - len(self.df)
        
        logger.info(f"Dropped {dropped_rows} rows with NaN values")
        logger.info(f"Final dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        return self.df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature column names (excluding target and metadata).
        
        Returns:
            List of feature column names
        """
        exclude_cols = ['Date', Config.TARGET_COLUMN]
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        return feature_cols
    
    def save_processed_data(self, ticker: str):
        """
        Save processed data with features.
        
        Args:
            ticker: Stock ticker symbol for filename
        """
        output_path = Config.get_processed_data_path(ticker)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")


def main():
    """Main function for testing feature engineering."""
    from src.data.data_loader import StockDataLoader
    
    # Load data
    loader = StockDataLoader(ticker=Config.DEFAULT_TICKER)
    raw_data_path = Config.get_raw_data_path(Config.DEFAULT_TICKER)
    
    if raw_data_path.exists():
        df = loader.load_from_file()
    else:
        df = loader.download_data()
        df = loader.clean_data()
        loader.save_data()
    
    # Feature engineering
    engineer = FeatureEngineer(df)
    processed_df = engineer.create_all_features()
    
    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"\nFeature columns: {engineer.get_feature_columns()}")
    print(f"\nFirst few rows:")
    print(processed_df.head(10))
    print(f"\nTarget distribution:")
    print(processed_df[Config.TARGET_COLUMN].value_counts())
    print(f"\nTarget balance: {processed_df[Config.TARGET_COLUMN].mean():.2%} positive")
    
    # Save
    engineer.save_processed_data(Config.DEFAULT_TICKER)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
