"""
End-to-end training pipeline.
Orchestrates data loading, feature engineering, training, and evaluation.
"""

import logging
from pathlib import Path
import sys

from src.data.data_loader import StockDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.train import StockTrendModel
from src.evaluation.evaluate import ModelEvaluator
from src.utils.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(ticker: str = Config.DEFAULT_TICKER, model_type: str = 'logistic'):
    """
    Run the complete training pipeline.
    
    Args:
        ticker: Stock ticker symbol
        model_type: 'logistic' or 'xgboost'
    """
    logger.info("="*70)
    logger.info(f"Starting StockFlowML Pipeline for {ticker}")
    logger.info("="*70)
    
    # Step 1: Data Loading
    logger.info("\n[1/5] Data Loading")
    logger.info("-" * 50)
    
    loader = StockDataLoader(ticker=ticker)
    raw_data_path = Config.get_raw_data_path(ticker)
    
    # Download or load existing data
    if raw_data_path.exists():
        logger.info("Loading existing raw data...")
        df_raw = loader.load_from_file()
    else:
        logger.info("Downloading new data from Yahoo Finance...")
        df_raw = loader.download_data()
        df_raw = loader.clean_data()
        loader.save_data()
    
    # Step 2: Feature Engineering
    logger.info("\n[2/5] Feature Engineering")
    logger.info("-" * 50)
    
    engineer = FeatureEngineer(df_raw)
    df_processed = engineer.create_all_features()
    engineer.save_processed_data(ticker)
    
    feature_cols = engineer.get_feature_columns()
    logger.info(f"Created {len(feature_cols)} features")
    
    # Step 3: Model Training
    logger.info(f"\n[3/5] Model Training ({model_type})")
    logger.info("-" * 50)
    
    model = StockTrendModel(model_type=model_type)
    X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df_processed)
    model.train(X_train, y_train)
    model.save_model()
    
    # Step 4: Evaluation
    logger.info("\n[4/5] Model Evaluation")
    logger.info("-" * 50)
    
    # Get predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_evaluator = ModelEvaluator(y_train, train_preds)
    train_metrics = train_evaluator.calculate_metrics()
    
    test_evaluator = ModelEvaluator(y_test, test_preds)
    test_metrics = test_evaluator.calculate_metrics()
    
    # Print summary
    logger.info(f"\nTraining Accuracy: {train_metrics['accuracy']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Step 5: Generate Report
    logger.info("\n[5/5] Generating Report")
    logger.info("-" * 50)
    
    dataset_info = {
        'ticker': ticker,
        'total_samples': len(df_processed),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'date_range': f"{df_processed['Date'].min().date()} to {df_processed['Date'].max().date()}"
    }
    
    test_evaluator.save_markdown_report(
        model_type=model_type.title(),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        dataset_info=dataset_info
    )
    
    # Final Summary
    logger.info("\n" + "="*70)
    logger.info("Pipeline Complete! ✓")
    logger.info("="*70)
    logger.info(f"Raw Data: {Config.get_raw_data_path(ticker)}")
    logger.info(f"Processed Data: {Config.get_processed_data_path(ticker)}")
    logger.info(f"Model: {Config.get_model_path(f'{model_type}_model')}")
    logger.info(f"Metrics: {Config.get_metrics_path()}")
    logger.info("="*70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockFlowML Training Pipeline')
    parser.add_argument(
        '--ticker',
        type=str,
        default=Config.DEFAULT_TICKER,
        help=f'Stock ticker symbol (default: {Config.DEFAULT_TICKER})'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='logistic',
        choices=['logistic', 'xgboost'],
        help='Model type to train (default: logistic)'
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(ticker=args.ticker, model_type=args.model)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
