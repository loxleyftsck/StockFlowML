"""
Model training module for stock trend prediction.
Implements baseline Logistic Regression with optional XGBoost.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
import logging
import json
from datetime import datetime

from src.utils.config import Config

logger = logging.getLogger(__name__)


class StockTrendModel:
    """Train and manage stock trend prediction models."""
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize model trainer.
        
        Args:
            model_type: 'logistic' or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
    
    def _create_model(self):
        """Create the specified model."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                random_state=Config.RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                random_state=Config.RANDOM_STATE,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=1.0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        test_size: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training.
        
        Args:
            df: Processed DataFrame with features and target
            test_size: Fraction for test set (default from Config)
            
        Returns:
            X_train, X_test, y_train, y_test, train_df, test_df
        """
        if test_size is None:
            test_size = 1 - Config.TRAIN_TEST_SPLIT
        
        # Exclude Date and target from features
        exclude_cols = ['Date', Config.TARGET_COLUMN]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Split data temporally (not random - important for time series)
        split_idx = int(len(df) * Config.TRAIN_TEST_SPLIT)
        train_df = df[:split_idx].copy()
        test_df = df[split_idx:].copy()
        
        X_train = train_df[feature_cols].values
        y_train = train_df[Config.TARGET_COLUMN].values
        X_test = test_df[feature_cols].values
        y_test = test_df[Config.TARGET_COLUMN].values
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train positive rate: {y_train.mean():.2%}")
        logger.info(f"Test positive rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test, train_df, test_df
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> 'StockTrendModel':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        logger.info("Training complete")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, model_name: str = None):
        """
        Save trained model and scaler.
        
        Args:
            model_name: Name for saved model (default: model_type)
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        if model_name is None:
            model_name = f"{self.model_type}_model"
        
        model_path = Config.get_model_path(model_name)
        
        # Save model and scaler together
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(save_dict, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str = None):
        """
        Load a saved model.
        
        Args:
            model_name: Name of saved model (default: model_type)
        """
        if model_name is None:
            model_name = f"{self.model_type}_model"
        
        model_path = Config.get_model_path(model_name)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        save_dict = joblib.load(model_path)
        
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.feature_names = save_dict['feature_names']
        self.model_type = save_dict['model_type']
        
        logger.info(f"Model loaded from {model_path}")


def main():
    """Main function for training."""
    from src.features.feature_engineering import FeatureEngineer
    
    # Load processed data
    processed_path = Config.get_processed_data_path(Config.DEFAULT_TICKER)
    
    if not processed_path.exists():
        print("Processed data not found. Run feature_engineering.py first.")
        return
    
    df = pd.read_csv(processed_path, parse_dates=['Date'])
    print(f"Loaded {len(df)} samples")
    
    # Train Logistic Regression
    model = StockTrendModel(model_type='logistic')
    X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
    model.train(X_train, y_train)
    
    # Quick evaluation
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()
    
    print(f"\nLogistic Regression Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model.save_model()
    
    print(f"\n✓ Model saved to {Config.MODELS_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
