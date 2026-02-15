"""
Feast Feature Definitions for StockFlowML

Defines feature views for stock technical indicators and price data.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, String, Int64

# Define stock entity (ticker symbol)
stock_entity = Entity(
    name="ticker",
    description="Stock ticker symbol (e.g., BBCA.JK)",
    join_keys=["ticker"]
)

# Define data source (parquet files from processed data)
stock_features_source = FileSource(
    name="stock_features_source",
    path="data/processed/feast_features.parquet",
    timestamp_field="Date",
)

# Feature View: Technical Indicators
stock_technical_features = FeatureView(
    name="stock_technical_features",
    entities=[stock_entity],
    ttl=timedelta(days=1),  # Features valid for 1 day
    schema=[
        Field(name="Open", dtype=Float64, description="Opening price"),
        Field(name="High", dtype=Float64, description="Highest price"),
        Field(name="Low", dtype=Float64, description="Lowest price"),
        Field(name="Close", dtype=Float64, description="Closing price"),
        Field(name="Volume", dtype=Float64, description="Trading volume"),
        Field(name="returns", dtype=Float64, description="Daily returns"),
        Field(name="ma_5", dtype=Float64, description="5-day moving average"),
        Field(name="ma_10", dtype=Float64, description="10-day moving average"),
        Field(name="ma_20", dtype=Float64, description="20-day moving average"),
        Field(name="volatility_5", dtype=Float64, description="5-day volatility"),
        Field(name="volatility_10", dtype=Float64, description="10-day volatility"),
        Field(name="volatility_20", dtype=Float64, description="20-day volatility"),
    ],
    source=stock_features_source,
    online=True,
    tags={"team": "ml", "project": "stockflowml"},
)
