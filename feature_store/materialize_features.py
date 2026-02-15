"""
Materialize Features to Feast Online Store

This script converts processed CSV data to Parquet format and materializes
features to the Feast online store for low-latency serving.
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_feast_data(ticker: str = "BBCA.JK") -> Path:
    """
    Convert processed CSV to Parquet format for Feast.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Path to created parquet file
    """
    logger.info(f"Preparing Feast data for {ticker}...")
    
    # Load processed data
    processed_path = project_root / "data" / "processed" / f"{ticker}_processed.csv"
    
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")
    
    df = pd.read_csv(processed_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add ticker column (entity key)
    df['ticker'] = ticker
    
    # Select feature columns
    feature_columns = [
        'Date', 'ticker',
        'Open', 'High', 'Low', 'Close', 'Volume',
        'returns', 'ma_5', 'ma_10', 'ma_20',
        'volatility_5', 'volatility_10', 'volatility_20'
    ]
    
    # Filter to only include available columns
    available_columns = [col for col in feature_columns if col in df.columns]
    df_feast = df[available_columns].copy()
    
    # Remove rows with NaN (from rolling windows)
    df_feast = df_feast.dropna()
    
    # Save as parquet
    output_path = project_root / "data" / "processed" / "feast_features.parquet"
    df_feast.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(df_feast)} rows to {output_path}")
    logger.info(f"Date range: {df_feast['Date'].min()} to {df_feast['Date'].max()}")
    
    return output_path


def materialize_features(start_date: datetime = None, end_date: datetime = None):
    """
    Materialize features to Feast online store.
    
    Args:
        start_date: Start date for materialization (default: 30 days ago)
        end_date: End date for materialization (default: now)
    """
    try:
        from feast import FeatureStore
    except ImportError:
        logger.error("Feast not installed. Run: pip install feast")
        return
    
    # Set default dates
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    logger.info(f"Materializing features from {start_date} to {end_date}...")
    
    # Initialize Feast store
    repo_path = project_root / "feature_store" / "feature_repo"
    store = FeatureStore(repo_path=str(repo_path))
    
    # Apply feature definitions
    logger.info("Applying feature definitions...")
    store.apply([])  # This reads from features.py
    
    # Materialize to online store
    logger.info("Materializing to online store...")
    store.materialize(
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info("✓ Materialization complete!")
    
    # Test retrieval
    logger.info("Testing feature retrieval...")
    entity_df = pd.DataFrame({
        "ticker": ["BBCA.JK"],
        "event_timestamp": [end_date]
    })
    
    features = store.get_online_features(
        features=[
            "stock_technical_features:Close",
            "stock_technical_features:returns",
            "stock_technical_features:ma_5",
            "stock_technical_features:volatility_5"
        ],
        entity_rows=entity_df.to_dict('records')
    ).to_dict()
    
    logger.info(f"Retrieved features: {features}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Materialize features to Feast")
    parser.add_argument("--ticker", default="BBCA.JK", help="Stock ticker")
    parser.add_argument("--days", type=int, default=30, help="Days to materialize")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare parquet, don't materialize")
    
    args = parser.parse_args()
    
    # Step 1: Prepare data
    try:
        parquet_path = prepare_feast_data(args.ticker)
        logger.info(f"✓ Data prepared: {parquet_path}")
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        sys.exit(1)
    
    # Step 2: Materialize (unless --prepare-only)
    if not args.prepare_only:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            materialize_features(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to materialize: {e}")
            logger.info("You can run with --prepare-only to just create the parquet file")
            sys.exit(1)
