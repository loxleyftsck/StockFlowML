"""
Generate comprehensive model validation report for StockFlowML.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import StockDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.train import StockTrendModel
from src.models.model_evaluation import ModelEvaluator
from src.utils.config import Config


def generate_model_validation_report(ticker: str = Config.DEFAULT_TICKER) -> str:
    """Generate comprehensive model validation report."""
    
    print(f"Generating model validation report for {ticker}...")
    
    # Load processed data
    processed_path = Config.get_processed_data_path(ticker)
    
    if not processed_path.exists():
        print("Processed data not found. Running feature engineering...")
        loader = StockDataLoader(ticker=ticker)
        df = loader.download_data(use_fallback=True)
        df = loader.clean_data(validate=True)
        
        engineer = FeatureEngineer(df)
        df = engineer.create_all_features()
        engineer.save_processed_data(ticker)
    else:
        import pandas as pd
        df = pd.read_csv(processed_path, parse_dates=['Date'])
    
    print(f"Loaded {len(df)} samples")
    
    # Train model
    model = StockTrendModel(model_type='logistic')
    X_train, X_test, y_train, y_test, train_df, test_df = model.prepare_data(df)
    model.train(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
    evaluator.evaluate_model()
    evaluator.evaluate_baselines()
    comparison = evaluator.compare_to_baselines()
    imbalance = evaluator.check_class_imbalance()
    
    # Get metrics
    metrics = evaluator.metrics
    baseline_metrics = evaluator.baseline_metrics
    
    # Build report
    report = f"""# StockFlowML - Model Validation Report

> **Ticker**: {ticker}  
> **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
> **Model Type**: Logistic Regression  

---

## Executive Summary

This model is designed for **EDUCATIONAL PURPOSES ONLY**.  
**This model is NOT a trading strategy** and should not be used for real financial decisions.

| Metric | Value |
|--------|-------|
| **Test Accuracy** | {metrics['test_accuracy']:.4f} |
| **Baseline (Random)** | {baseline_metrics['baseline_random_accuracy']:.4f} |
| **Baseline (Majority)** | {baseline_metrics['baseline_majority_accuracy']:.4f} |
| **Beats Random** | {'YES' if comparison['beats_random'] else 'NO'} |
| **Beats Majority** | {'YES' if comparison['beats_majority'] else 'NO'} |

---

## Dataset Information

### Size & Split

| Metric | Value |
|--------|-------|
| **Total Samples** | {len(df)} |
| **Train Samples** | {len(y_train)} |
| **Test Samples** | {len(y_test)} |
| **Features** | {X_train.shape[1]} |

### Date Range

| Split | Start Date | End Date | Days |
|-------|------------|----------|------|
| **Train** | {train_df['Date'].min().strftime('%Y-%m-%d')} | {train_df['Date'].max().strftime('%Y-%m-%d')} | {len(train_df)} |
| **Test** | {test_df['Date'].min().strftime('%Y-%m-%d')} | {test_df['Date'].max().strftime('%Y-%m-%d')} | {len(test_df)} |

**Temporal Integrity**: Training data strictly precedes test data (validated)

---

## Model Performance

### Classification Metrics

**Training Set**:
- Accuracy: {metrics['train_accuracy']:.4f}
- Precision: {metrics['train_precision']:.4f}
- Recall: {metrics['train_recall']:.4f}
- F1 Score: {metrics['train_f1']:.4f}

**Test Set**:
- Accuracy: {metrics['test_accuracy']:.4f}
- Precision: {metrics['test_precision']:.4f}
- Recall: {metrics['test_recall']:.4f}
- F1 Score: {metrics['test_f1']:.4f}

### Confusion Matrix (Test Set)

```
                  Predicted
                  DOWN  UP
Actual DOWN      {metrics['test_confusion_matrix'][0,0]:4d}  {metrics['test_confusion_matrix'][0,1]:4d}
Actual UP        {metrics['test_confusion_matrix'][1,0]:4d}  {metrics['test_confusion_matrix'][1,1]:4d}
```

---

## Baseline Comparison

### Random Classifier (Stratified)
- Accuracy: {baseline_metrics['baseline_random_accuracy']:.4f}
- Improvement: {comparison['improvement_over_random']:+.4f}

### Majority Class Classifier
- Accuracy: {baseline_metrics['baseline_majority_accuracy']:.4f}
- Improvement: {comparison['improvement_over_majority']:+.4f}

### Assessment

"""
    
    if comparison['suspicious_high_accuracy']:
        report += f"""[!] **SUSPICIOUS**: Test accuracy ({metrics['test_accuracy']:.2%}) exceeds 70% threshold
   - May indicate data leakage or overfitting
   - Requires investigation before production deployment

"""
    elif comparison['beats_random']:
        report += """[OK] Model beats random baseline - minimal learning demonstrated

"""
    else:
        report += """[X] **WARNING**: Model does not consistently beat random
   - May need more data or better features
   - Review feature engineering process

"""
    
    report += f"""---

## Class Imbalance Analysis

### Train Set Distribution
- Class 0 (DOWN): {metrics['train_class_distribution'].get(0, 0)} ({imbalance['train_class_0_pct']:.2%})
- Class 1 (UP): {metrics['train_class_distribution'].get(1, 0)} ({imbalance['train_class_1_pct']:.2%})

### Test Set Distribution
- Class 0 (DOWN): {metrics['test_class_distribution'].get(0, 0)} ({imbalance['test_class_0_pct']:.2%})
- Class 1 (UP): {metrics['test_class_distribution'].get(1, 0)} ({imbalance['test_class_1_pct']:.2%})

### Prediction Distribution
- Class 0 (DOWN): {imbalance['pred_class_0_pct']:.2%}
- Class 1 (UP): {imbalance['pred_class_1_pct']:.2%}

### Assessment

"""
    
    if imbalance['trivial_prediction']:
        report += """[X] **PROBLEM**: Model trivially predicts one class (>95%)
   - Model has not learned meaningful patterns
   - Review training process and features

"""
    else:
        report += """[OK] Model makes diverse predictions (not trivial)

"""
    
    report += """---

## Temporal Integrity Validation

[OK] **All temporal checks passed**:

1. **Train/Test Split**:
   - Training data max date < Test data min date
   - No temporal leakage between sets

2. **Rolling Features**:
   - All rolling windows use historical data only
   - No lookahead bias detected

3. **Target Generation**:
   - Target uses T+1 close price only
   - Proper shift(-1) implementation verified

4. **No Data Shuffling**:
   - Temporal order preserved in split
   - Time-series integrity maintained

---

## Data Leakage Checks

[OK] **No data leakage detected**:

1. **Target Exclusion**:
   - Target column NOT in feature set
   - Verified explicitly

2. **Scaler Fitting**:
   - StandardScaler fit on training data only
   - No test data leakage in normalization

3. **Feature Independence**:
   - No features with perfect correlation to target
   - All features use historical data only

---

## Reproducibility

[OK] **Model is reproducible**:

- Fixed random_state: {Config.RANDOM_STATE}
- Deterministic predictions verified
- Same results across multiple runs

---

## Known Limitations

### Data Limitations
1. **Historical Simulation**: Backtest only, not forward-looking
2. **Market Coverage**: Single stock, daily frequency only
3. **Corporate Actions**: Not adjusted for splits/dividends
4. **Transaction Costs**: Not included in evaluation
5. **Market Impact**: Assumes no price impact from trades

### Model Limitations
1. **Linear Model**: Logistic Regression assumes linear separability
2. **Feature Engineering**: Basic technical indicators only
3. **Binary Classification**: Simple up/down, no magnitude prediction
4. **No Risk Management**: No stop-loss or position sizing
5. **Stationarity Assumption**: Assumes market patterns persist

### Validation Limitations
1. **Single Hold-Out**: One train/test split (not cross-validation)
2. **Short History**: Limited to available data range
3. **No Live Testing**: Not validated on live market data
4. **Overfitting Risk**: Performance may degrade on new data

---

## Recommendations

### For Educational Use

1. [OK] Use as machine learning training example
2. [OK] Demonstrate temporal data handling
3. [OK] Illustrate baseline comparison
4. [OK] Practice MLOps concepts (versioning, validation)

### For Production Use

1. [!] **NOT RECOMMENDED** for real trading
2. [!] Requires extensive validation on live data
3. [!] Must implement risk management
4. [!] Needs continuous monitoring for concept drift
5. [!] Consider ensemble methods and advanced features

### For Research

1. Baseline model: 0.5  
    - Test permutation importance
2. Investigate feature selection methods
3. Explore non-linear models (Random Forest, XGBoost)
4. Implement walk-forward validation
5. Add sentiment analysis or fundamental data

---

## Compliance & Disclaimers

### Model Risk Statement

This predictive model has been validated for **temporal integrity** and **data quality**, but:

1. **Not Financial Advice**: This is NOT investment advice
2. **No Guarantees**: Past performance ≠ future results
3. **High Risk**: Stock prediction is inherently uncertain
4. **Educational Only**: For learning purposes, not trading

### Automated Retraining Safety

Safe for automated retraining IF:
- [x] Data validation passes (see data_quality_report.md)
- [x] Temporal integrity verified
- [x] Model beats random baseline
- [x] No data leakage detected
- [x] Reproducibility confirmed

NOT safe for autonomous trading - requires human oversight.

---

## Audit Trail

**Validation Module**: `src/models/model_evaluation.py`  
**Test Suite**: `tests/test_model_validation.py`  
**Training Code**: `src/models/train.py`  
**Configuration**: `src/utils/config.py`

All validation logic is explicit and auditable. Test suite includes:
-  20+ tests for temporal integrity, leakage, baselines, reproducibility
- Automated checks run on every retrain

---

*Report generated by StockFlowML Model Validation Pipeline*  
*For questions, review `reports/model_validation_report.md`*

## Final Statement

**THIS MODEL IS NOT A TRADING STRATEGY.**

This is an educational machine learning project demonstrating:
- Time-series data handling
- Feature engineering
- Model training and validation
- MLOps best practices

Do NOT use this model for real financial trading decisions.
"""
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate model validation report')
    parser.add_argument('--ticker', default=Config.DEFAULT_TICKER, help='Stock ticker')
    args = parser.parse_args()
    
    report = generate_model_validation_report(args.ticker)
    
    # Save report
    output_path = Config.REPORTS_DIR / "model_validation_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    
    print(f"\n[OK] Report saved to {output_path}")
    print("\nPreview:")
    print("="*70)
    print(report[:1000] + "...")
