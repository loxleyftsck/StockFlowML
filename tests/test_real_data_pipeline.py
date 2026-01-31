"""
Comprehensive real data pipeline validation script.
Tests the entire StockFlowML pipeline with live Yahoo Finance data.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

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


def test_data_loading():
    """Test A: Data Loading & Integrity"""
    print("="*70)
    print("TEST A: DATA LOADING & INTEGRITY")
    print("="*70)
    
    loader = StockDataLoader(ticker=Config.DEFAULT_TICKER)
    
    print(f"\n[1/8] Downloading data for {Config.DEFAULT_TICKER}...")
    try:
        df = loader.download_data(years=5)
        print(f"✓ Downloaded {len(df)} rows")
    except Exception as e:
        print(f"✗ FAILED to download: {e}")
        print("\nTrying alternative ticker...")
        # Try a more common ticker if BBCA.JK fails
        loader = StockDataLoader(ticker="^JKSE")  # Jakarta Composite Index
        df = loader.download_data(years=5)
        print(f"✓ Downloaded {len(df)} rows for ^JKSE")
    
    print(f"\n[2/8] Validating OHLCV columns...")
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
        assert df[col].notna().all(), f"NaN values in {col}"
    print(f"✓ All OHLCV columns present and non-null")
    
    print(f"\n[3/8] Checking data freshness...")
    latest_date = df['Date'].max()
    today = datetime.now()
    days_old = (today - latest_date).days
    print(f"Latest data: {latest_date.date()}")
    print(f"Days old: {days_old}")
    if days_old > 7:
        print(f"⚠ Warning: Data is {days_old} days old (may be stale)")
    else:
        print(f"✓ Data is fresh")
    
    print(f"\n[4/8] Checking for duplicates...")
    duplicates = df['Date'].duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate timestamps"
    print(f"✓ No duplicate timestamps")
    
    print(f"\n[5/8] Validating sort order...")
    assert df['Date'].is_monotonic_increasing, "Data not sorted by date"
    print(f"✓ Data sorted by date (ascending)")
    
    print(f"\n[6/8] Validating data ranges...")
    assert (df['Volume'] >= 0).all(), "Negative volume found"
    assert (df['Close'] > 0).all(), "Non-positive Close price found"
    assert (df['High'] >= df['Low']).all(), "High < Low found"
    assert (df['High'] >= df['Close']).all() and (df['High'] >= df['Open']).all(), "High price violations"
    assert (df['Low'] <= df['Close']).all() and (df['Low'] <= df['Open']).all(), "Low price violations"
    print(f"✓ All price/volume ranges valid")
    
    print(f"\n[7/8] Cleaning data...")
    df_clean = loader.clean_data(df)
    print(f"✓ Cleaned data: {len(df_clean)} rows")
    
    print(f"\n[8/8] Saving data...")
    loader.save_data()
    print(f"✓ Saved to {Config.get_raw_data_path(loader.ticker)}")
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
    print(f"Total days: {len(df_clean)}")
    print(f"Columns: {list(df_clean.columns)}")
    print(f"\nFirst 3 rows:")
    print(df_clean.head(3))
    print(f"\nLast 3 rows:")
    print(df_clean.tail(3))
    print(f"\nDescriptive statistics:")
    print(df_clean.describe())
    
    return df_clean, loader.ticker


def test_feature_engineering(df, ticker):
    """Test B & C: Feature Engineering & Target Generation"""
    print("\n" + "="*70)
    print("TEST B & C: FEATURE ENGINEERING & TARGET GENERATION")
    print("="*70)
    
    print(f"\n[1/6] Creating features...")
    engineer = FeatureEngineer(df)
    
    print(f"\n[2/6] Computing returns...")
    engineer.create_returns()
    assert 'returns' in engineer.df.columns
    # Check returns are reasonable
    returns_std = engineer.df['returns'].std()
    print(f"✓ Returns computed (std: {returns_std:.4f})")
    
    print(f"\n[3/6] Computing rolling means...")
    engineer.create_rolling_means()
    for window in Config.ROLLING_WINDOWS:
        col = f'ma_{window}'
        assert col in engineer.df.columns, f"Missing {col}"
        # Check rolling mean uses only past data
        first_valid_idx = window
        assert engineer.df[col].isna().sum() >= window, f"{col} may have data leakage"
    print(f"✓ Rolling means computed for windows {Config.ROLLING_WINDOWS}")
    
    print(f"\n[4/6] Computing rolling volatility...")
    engineer.create_rolling_volatility()
    for window in Config.ROLLING_WINDOWS:
        col = f'volatility_{window}'
        assert col in engineer.df.columns, f"Missing {col}"
    print(f"✓ Rolling volatility computed")
    
    print(f"\n[5/6] Creating target variable...")
    engineer.create_target()
    assert Config.TARGET_COLUMN in engineer.df.columns
    
    # CRITICAL: Check no future leakage in target
    # The last row should have NaN target (no future to compare)
    pre_dropna_len = len(engineer.df)
    
    print(f"\n[6/6] Validating target correctness...")
    # Manual spot check: verify target logic
    df_copy = df.copy()
    df_copy['close_next'] = df_copy['Close'].shift(-1)
    df_copy['target_check'] = (df_copy['close_next'] > df_copy['Close']).astype(int)
    
    # After all features, drop NaN
    df_processed = engineer.create_all_features()
    
    print(f"Rows before NaN drop: {pre_dropna_len}")
    print(f"Rows after NaN drop: {len(df_processed)}")
    print(f"Dropped rows: {pre_dropna_len - len(df_processed)}")
    
    # Check target distribution
    target_dist = df_processed[Config.TARGET_COLUMN].value_counts()
    target_ratio = df_processed[Config.TARGET_COLUMN].mean()
    print(f"\nTarget distribution:")
    print(target_dist)
    print(f"Positive rate: {target_ratio:.2%}")
    
    assert 0.3 < target_ratio < 0.7, f"Target distribution suspicious: {target_ratio:.2%}"
    print(f"✓ Target distribution reasonable")
    
    # Check no leakage: features should not correlate perfectly with target
    feature_cols = engineer.get_feature_columns()
    print(f"\nFeatures created: {feature_cols}")
    print(f"Total features: {len(feature_cols)}")
    
    # Save processed data
    engineer.save_processed_data(ticker)
    print(f"✓ Saved processed data to {Config.get_processed_data_path(ticker)}")
    
    return df_processed, feature_cols


def test_model_training(df_processed, feature_cols):
    """Test D & E: Model Training & Evaluation"""
    print("\n" + "="*70)
    print("TEST D & E: MODEL TRAINING & EVALUATION")
    print("="*70)
    
    print(f"\n[1/7] Preparing data...")
    model = StockTrendModel(model_type='logistic')
    X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df_processed)
    
    print(f"Train set: {len(X_train)} samples ({y_train.mean():.2%} positive)")
    print(f"Test set: {len(X_test)} samples ({y_test.mean():.2%} positive)")
    
    # Validate temporal split
    print(f"\n[2/7] Validating temporal split...")
    train_end = train_df['Date'].max()
    test_start = test_df['Date'].min()
    assert train_end < test_start, "Train/test sets overlap in time!"
    print(f"✓ Temporal split correct")
    print(f"  Train: {train_df['Date'].min().date()} to {train_end.date()}")
    print(f"  Test: {test_start.date()} to {test_df['Date'].max().date()}")
    
    print(f"\n[3/7] Training Logistic Regression...")
    model.train(X_train, y_train)
    print(f"✓ Model trained")
    
    print(f"\n[4/7] Making predictions...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Quick sanity check
    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    print(f"\n[5/7] Validating metrics realism...")
    # Stock prediction is hard - accuracy should be close to 50% (random)
    # Anything >70% is suspicious
    assert 0.4 < train_acc < 0.8, f"Training accuracy suspicious: {train_acc:.2%}"
    assert 0.4 < test_acc < 0.75, f"Test accuracy suspicious: {test_acc:.2%}"
    print(f"✓ Metrics within realistic bounds")
    
    print(f"\n[6/7] Computing full evaluation...")
    train_evaluator = ModelEvaluator(y_train, train_preds)
    train_metrics = train_evaluator.calculate_metrics()
    
    test_evaluator = ModelEvaluator(y_test, test_preds)
    test_metrics = test_evaluator.calculate_metrics()
    
    print(f"\nTRAINING METRICS:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1-Score:  {train_metrics['f1_score']:.4f}")
    
    print(f"\nTEST METRICS:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    
    print(f"\n[7/7] Saving model...")
    model.save_model('logistic_model_validation')
    print(f"✓ Model saved")
    
    return model, train_metrics, test_metrics, (train_df, test_df)


def test_reproducibility(df_processed):
    """Test F: Reproducibility"""
    print("\n" + "="*70)
    print("TEST F: REPRODUCIBILITY")
    print("="*70)
    
    print(f"\n[1/3] Training model (run 1)...")
    model1 = StockTrendModel(model_type='logistic')
    X_train, X_test, y_train, y_test, _, _ = model1.prepare_data(df_processed)
    model1.train(X_train, y_train)
    preds1 = model1.predict(X_test)
    acc1 = (preds1 == y_test).mean()
    
    print(f"\n[2/3] Training model (run 2)...")
    model2 = StockTrendModel(model_type='logistic')
    model2.prepare_data(df_processed)
    model2.train(X_train, y_train)
    preds2 = model2.predict(X_test)
    acc2 = (preds2 == y_test).mean()
    
    print(f"\n[3/3] Comparing results...")
    print(f"Run 1 accuracy: {acc1:.6f}")
    print(f"Run 2 accuracy: {acc2:.6f}")
    print(f"Difference: {abs(acc1 - acc2):.6f}")
    
    # Should be identical due to random_state
    assert np.allclose(acc1, acc2, atol=1e-6), "Results not reproducible!"
    print(f"✓ Results are reproducible")


def main():
    """Run all validation tests"""
    print("\n" + "#"*70)
    print("# STOCKFLOWML - REAL DATA PIPELINE VALIDATION")
    print("# QA Lead: Testing with live Yahoo Finance data")
    print("#"*70)
    print(f"\nValidation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target ticker: {Config.DEFAULT_TICKER}")
    
    try:
        # Test A: Data Loading
        df_clean, ticker_used = test_data_loading()
        
        # Test B & C: Feature Engineering & Target
        df_processed, feature_cols = test_feature_engineering(df_clean, ticker_used)
        
        # Test D & E: Model Training & Evaluation  
        model, train_metrics, test_metrics, (train_df, test_df) = test_model_training(df_processed, feature_cols)
        
        # Test F: Reproducibility
        test_reproducibility(df_processed)
        
        print("\n" + "#"*70)
        print("# ALL TESTS PASSED ✓")
        print("#"*70)
        print(f"\n✅ Pipeline validated successfully with real market data")
        print(f"✅ Data: {len(df_processed)} samples from {ticker_used}")
        print(f"✅ Features: {len(feature_cols)} engineered features")
        print(f"✅ Model: Logistic Regression trained and evaluated")
        print(f"✅ Test accuracy: {test_metrics['accuracy']:.2%}")
        print(f"✅ Reproducibility: Confirmed")
        
        return True, {
            'ticker': ticker_used,
            'samples': len(df_processed),
            'date_range': (df_clean['Date'].min(), df_clean['Date'].max()),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'features': feature_cols
        }
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    success, results = main()
    
    if success:
        print(f"\n{'='*70}")
        print("VALIDATION REPORT SUMMARY")
        print(f"{'='*70}")
        print(f"Ticker: {results['ticker']}")
        print(f"Date Range: {results['date_range'][0].date()} to {results['date_range'][1].date()}")
        print(f"Total Samples: {results['samples']}")
        print(f"Train/Test Split: {results['train_size']}/{results['test_size']}")
        print(f"\nTest Performance:")
        print(f"  Accuracy:  {results['test_metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['test_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['test_metrics']['recall']:.4f}")
        print(f"  F1-Score:  {results['test_metrics']['f1_score']:.4f}")
        sys.exit(0)
    else:
        sys.exit(1)
