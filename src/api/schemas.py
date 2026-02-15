from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class PredictionInput(BaseModel):
    """
    Input schema for single prediction request.
    Expects pre-computed features.
    """
    Open: float = Field(..., description="Opening price")
    High: float = Field(..., description="Highest price")
    Low: float = Field(..., description="Lowest price")
    Close: float = Field(..., description="Closing price")
    Volume: float = Field(..., description="Trading volume")
    
    # Engineered features
    returns: float = Field(..., description="Daily return")
    ma_5: float = Field(..., description="5-day Moving Average")
    ma_10: float = Field(..., description="10-day Moving Average")
    ma_20: float = Field(..., description="20-day Moving Average")
    volatility_5: float = Field(..., description="5-day Volatility")
    volatility_10: float = Field(..., description="10-day Volatility")
    volatility_20: float = Field(..., description="20-day Volatility")

    class Config:
        json_schema_extra = {
            "example": {
                "Open": 4500.0,
                "High": 4550.0,
                "Low": 4480.0,
                "Close": 4520.0,
                "Volume": 10000000.0,
                "returns": 0.004,
                "ma_5": 4480.0,
                "ma_10": 4470.0,
                "ma_20": 4450.0,
                "volatility_5": 0.01,
                "volatility_10": 0.012,
                "volatility_20": 0.015
            }
        }

class PredictionOutput(BaseModel):
    """
    Output schema for prediction response.
    """
    prediction: int = Field(..., description="Predicted class (0: Down, 1: Up)")
    probability: float = Field(..., description="Prediction probability for class 1")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = Field("unknown", description="Version of the model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class TickerPredictionInput(BaseModel):
    """
    Input schema for ticker-based prediction.
    Fetches features from Feast feature store.
    """
    ticker: str = Field(..., description="Stock ticker symbol (e.g., BBCA.JK)")
    timestamp: Optional[datetime] = Field(None, description="Timestamp for feature retrieval (default: now)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "BBCA.JK",
                "timestamp": "2024-01-15T10:00:00"
            }
        }

class HealthCheck(BaseModel):
    """Health check response schema."""
    status: str = "ok"
    version: str = "1.0.0"
    model_loaded: bool = False
    feast_enabled: bool = False
