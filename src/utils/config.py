"""
Configuration management for StockFlowML.
Provides centralized access to project settings and paths.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Centralized configuration manager."""
    
    # Project root
    ROOT_DIR = Path(__file__).parent.parent.parent
    
    # Data directories
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model directories
    MODELS_DIR = ROOT_DIR / "models"
    
    # Reports directory
    REPORTS_DIR = ROOT_DIR / "reports"
    
    # Default stock ticker
    DEFAULT_TICKER = "BBCA.JK"
    
    # Data settings
    LOOKBACK_YEARS = 5
    TRAIN_TEST_SPLIT = 0.8
    
    # Feature engineering
    ROLLING_WINDOWS = [5, 10, 20]
    
    # Model settings
    RANDOM_STATE = 42
    
    # Target definition
    # Binary: 1 if Close[t+1] > Close[t], else 0
    TARGET_COLUMN = "target"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_raw_data_path(cls, ticker: str) -> Path:
        """Get path for raw data file."""
        return cls.RAW_DATA_DIR / f"{ticker}_raw.csv"
    
    @classmethod
    def get_processed_data_path(cls, ticker: str) -> Path:
        """Get path for processed data file."""
        return cls.PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get path for saved model."""
        return cls.MODELS_DIR / f"{model_name}.pkl"
    
    @classmethod
    def get_metrics_path(cls) -> Path:
        """Get path for metrics report."""
        return cls.REPORTS_DIR / "metrics.md"


# Ensure directories exist on import
Config.ensure_directories()
