"""
Generate data feasibility report for StockFlowML.
Production-grade assessment of data quality and readiness.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import StockDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.data.data_feasibility import validate_data_feasibility, DataFeasibilityError
from src.utils.config import Config


def generate_feasibility_report(ticker: str = Config.DEFAULT_TICKER) -> str:
    """Generate markdown feasibility report."""
    
    print(f"Generating data feasibility report for {ticker}...")
    
    # Load processed data (with features and target)
    processed_path = Config.get_processed_data_path(ticker)
    
    if not processed_path.exists():
        print("Processed data not found. Running full pipeline...")
        loader = StockDataLoader(ticker=ticker)
        df = loader.download_data(use_fallback=True)
        df = loader.clean_data(validate=True)
        
        engineer = FeatureEngineer(df)
        df = engineer.create_all_features()
        engineer.save_processed_data(ticker)
    else:
        df = pd.read_csv(processed_path, parse_dates=['Date'])
    
    print(f"Loaded {len(df)} samples")
    
    # Run feasibility validation
    try:
        summary = validate_data_feasibility(
            df, 
            ticker=ticker,
            target_col=Config.TARGET_COLUMN
        )
        verdict = "PASS"
        production_ready = True
    except DataFeasibilityError as e:
        # Capture failure details
        summary = {
            'verdict': 'FAIL',
            'error_message': str(e),
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'production_ready': False
        }
        verdict = "FAIL"
        production_ready = False
        print(f"\n⚠️  FEASIBILITY CHECK FAILED:")
        print(f"  {e}\n")
    
    # Build report
    report = f"""# StockFlowML - Data Feasibility Report

> **Ticker**: {ticker}  
> **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
> **Verdict**: **{verdict}**  
> **Production Ready**: {'✓ YES' if production_ready else '✗ NO'}

---

## Executive Summary

This report assesses whether the dataset meets minimum production quality standards for machine learning training.

**Verdict**: {('[OK] PRODUCTION-READY' if verdict == 'PASS' else '[FAIL] NOT PRODUCTION-READY')}

---

## Feasibility Criteria

All checks must PASS for production deployment. Any FAIL triggers hard error.

### 1. Minimum Samples

**Criterion**: Dataset must contain >= 100 trading days  
**Rationale**: Statistical validity requires sufficient samples for train/test split

**Threshold**: `MIN_SAMPLES = 100`

"""
    
    if verdict == "PASS":
        # Add detailed passing checks
        checks = summary.get('check_details', {})
        
        # Minimum samples
        min_samples = checks.get('minimum_samples', {})
        report += f"""**Status**: ✓ PASS  
**Value**: {min_samples.get('n_samples', 'N/A')} samples  

### 2. Data Completeness

**Criterion**: >= 90% of OHLCV data must be present  
**Rationale**: > 10% missing data indicates poor data quality

**Threshold**: `MIN_COMPLETENESS_PCT = 90.0%`

"""
        completeness = checks.get('completeness', {})
        report += f"""**Status**: ✓ PASS  
**Value**: {completeness.get('completeness_pct', 0):.2f}% complete  
**Missing Cells**: {completeness.get('missing_cells', 0)} out of {completeness.get('total_cells', 0)}

### 3. Temporal Continuity

**Criterion**: No gaps > 10 trading days  
**Rationale**: Large gaps indicate missing data or market closures

**Threshold**: `MAX_GAP_DAYS = 10`

"""
        temporal = checks.get('temporal_continuity', {})
        report += f"""**Status**: ✓ PASS  
**Max Gap**: {temporal.get('max_gap_days', 0)} days  
**Large Gaps Found**: {temporal.get('n_large_gaps', 0)}

### 4. Outlier Detection

**Criterion**: < 1% of returns should be extreme outliers  
**Rationale**: Excessive outliers suggest data quality issues  
**Method**: Z-score on daily returns

**Threshold**: `OUTLIER_THRESHOLD = 4.0 (z-score)`

"""
        outliers = checks.get('outliers', {})
        report += f"""**Status**: ✓ PASS  
**Outliers Found**: {outliers.get('n_outliers', 0)} ({outliers.get('outlier_pct', 0):.2f}%)  
**Z-score Threshold**: {outliers.get('z_threshold', 4.0)}

### 5. Label Balance

**Criterion**: Minority class between 20% and 80%  
**Rationale**: Severely imbalanced labels make classification difficult

**Thresholds**: `MIN_LABEL_BALANCE = 20%`, `MAX_LABEL_BALANCE = 80%`

"""
        label = checks.get('label_balance', {})
        if label:
            report += f"""**Status**: ✓ PASS  
**Minority Ratio**: {label.get('minority_ratio', 0):.2%}  
**Distribution**: {label.get('label_distribution', {})}

### 6. Look-Ahead Bias Check

**Criterion**: No features with future information  
**Rationale**: Target leakage invalidates model

"""
            lookahead = checks.get('lookahead_bias', {})
            report += f"""**Status**: ✓ PASS  
**Suspicious Columns**: None

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | {summary.get('n_samples', 0)} |
| **Completeness** | {completeness.get('completeness_pct', 0):.2f}% |
| **Max Gap** | {temporal.get('max_gap_days', 0)} days |
| **Outliers** | {outliers.get('n_outliers', 0)} ({outliers.get('outlier_pct', 0):.2f}%) |
| **Label Balance** | {label.get('minority_ratio', 0):.2%} minority class |

---

## Production Readiness Assessment

**Status**: ✓ **PRODUCTION-READY**

All feasibility checks PASSED. Data meets minimum quality standards for ML training.

### Approved Uses
- Model training and validation
- Backtesting strategies
- Feature engineering experiments
- Educational demonstrations

### Known Limitations
1. **Historical Data Only**: This is backtested data, not live
2. **Single Ticker**: Analysis limited to {ticker}
3. **Daily Frequency**: No intraday data
4. **No Corporate Actions**: Splits/dividends may affect continuity
5. **Market Conditions**: Past patterns may not persist

---

## Recommendations

✓ **Proceed with model training**  
✓ Data quality is acceptable for production ML pipeline  
✓ Continue to monitor for drift in future retraining cycles

---

## Audit Trail

**Validation Module**: `src/data/data_feasibility.py`  
**Test Suite**: `tests/test_data_feasibility.py`  
**Generated By**: `scripts/generate_feasibility_report.py`

All checks explicit, logged, and auditable.

---

*Report generated by StockFlowML Data Feasibility Pipeline*
"""
    else:
        # FAIL case - minimal report
        error_msg = summary.get('error_message', 'Unknown error')
        report += f"""**Status**: ✗ FAIL

---

## Failure Details

```
{error_msg}
```

---

## Production Readiness Assessment

**Status**: ✗ **NOT PRODUCTION-READY**

Data FAILED feasibility checks. Cannot proceed with model training.

### Required Actions

1. Review error message above
2. Fix data quality issues at source
3. Re-run feasibility validation
4. Only proceed when all checks PASS

---

## Audit Trail

**Validation Module**: `src/data/data_feasibility.py`  
**Test Suite**: `tests/test_data_feasibility.py`  
**Generated By**: `scripts/generate_feasibility_report.py`

---

*Report generated by StockFlowML Data Feasibility Pipeline*
"""
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate data feasibility report')
    parser.add_argument('--ticker', default=Config.DEFAULT_TICKER, help='Stock ticker')
    args = parser.parse_args()
    
    report = generate_feasibility_report(args.ticker)
    
    # Save report
    output_path = Config.REPORTS_DIR / "data_feasibility.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding='utf-8')
    
    print(f"\n✓ Report saved to {output_path}")
    print("\nPreview:")
    print("="*70)
    print(report[:500] + "...")
