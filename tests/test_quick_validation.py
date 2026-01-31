"""
Simplified validation test using a known-working ticker.
Tests the StockFlowML pipeline with AAPL as fallback to verify pipeline logic.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import StockDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.train import StockTrendModel
from src.evaluation.evaluate import ModelEvaluator
from src.utils.config import Config

import pandas as pd
import numpy as np
from datetime import datetime


def run_quick_validation():
    """Run quick validation with a working ticker"""
    print("="*70)
    print("STOCKFLOWML - QUICK VALIDATION TEST")
    print("="*70)
    
    # Use AAPL as it's reliably available on Yahoo Finance
    test_ticker = "AAPL"
    print(f"\nUsing ticker: {test_ticker} (fallback from {Config.DEFAULT_TICKER})")
    print("Note: Testing pipeline logic, not Indonesian market data")
    
    # Step 1: Download
    print("\n[1/5] Downloading data...")
    loader = StockDataLoader(ticker=test_ticker)
    df = loader.download_data(years=5)
    df = loader.clean_data()
    print(f"✓ Downloaded and cleaned {len(df)} rows")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Step 2: Feature Engineering
    print("\n[2/5] Feature engineering...")
    engineer = FeatureEngineer(df)
    df_processed = engineer.create_all_features()
    print(f"✓ Created {len(engineer.get_feature_columns())} features")
    print(f"  Processed: {len(df_processed)} rows after NaN drop")
    print(f"  Target distribution: {df_processed[Config.TARGET_COLUMN].mean():.2%} positive")
    
    # Step 3: Train/Test Split
    print("\n[3/5] Preparing train/test split...")
    model = StockTrendModel(model_type='logistic')
    X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df_processed)
    print(f"✓ Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Train end: {train_df['Date'].max().date()}")
    print(f"  Test start: {test_df['Date'].min().date()}")
    
    # Step 4: Training
    print("\n[4/5] Training model...")
    model.train(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    print(f"✓ Model trained")
    
    # Step 5: Evaluation
    print("\n[5/5] Evaluating...")
    test_evaluator = ModelEvaluator(y_test, test_preds)
    test_metrics = test_evaluator.calculate_metrics()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {test_metrics.get('true_negatives', 'N/A')}, FP: {test_metrics.get('false_positives', 'N/A')}")
    print(f"  FN: {test_metrics.get('false_negatives', 'N/A')}, TP: {test_metrics.get('true_positives', 'N/A')}")
    
    print(f"\n{'='*70}")
    print("✅ PIPELINE VALIDATION SUCCESSFUL")
    print(f"{'='*70}")
    print("Pipeline logic verified with real market data.")
    print(f"Note: For {Config.DEFAULT_TICKER}, ensure Yahoo Finance access or use alternative data source.")
    
    return test_metrics


if __name__ == "__main__":
    try:
        metrics = run_quick_validation()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
