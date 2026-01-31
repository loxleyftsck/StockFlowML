"""
Generate larger synthetic dataset for feasibility demonstration.
Creates production-quality synthetic stock data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def generate_synthetic_stock_data(ticker="DEMO", n_days=500, start_price=10000):
    """Generate realistic synthetic stock data."""
    np.random.seed(42)
    
    # Generate business days
    dates = pd.bdate_range(start='2020-01-01', periods=n_days, freq='B')
    
    # Generate price path using geometric Brownian motion
    returns = np.random.normal(0.0005, 0.02, n_days)  # Mean 0.05% daily, 2% vol
    price_path = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    close_prices = price_path
    open_prices = close_prices * np.random.uniform(0.99, 1.01, n_days)
    high_prices = np.maximum(open_prices, close_prices) * np.random.uniform(1.00, 1.02, n_days)
    low_prices = np.minimum(open_prices, close_prices) * np.random.uniform(0.98, 1.00, n_days)
    
    # Ensure OHLC relationships
    high_prices = np.maximum.reduce([open_prices, high_prices, close_prices])
    low_prices = np.minimum.reduce([open_prices, low_prices, close_prices])
    
    # Generate volume with some correlation to price changes
    base_volume = 20000000
    volume_multiplier = 1 + np.abs(returns) * 5  # Higher volume on big moves
    volumes = (base_volume * volume_multiplier).astype(int)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_stock_data(ticker="DEMO", n_days=500)
    
    # Save to raw data directory
    output_path = Path("data/raw/DEMO_raw.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} days of synthetic data")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"  Saved to: {output_path}")
