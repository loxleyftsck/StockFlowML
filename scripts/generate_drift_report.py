"""
Generate Drift Detection Report - Windows Compatible Version

This script compares reference (baseline) and current data to detect
data drift and target drift using Evidently AI.

Usage:
    python scripts/generate_drift_report.py --reference data/processed/BBCA.JK_processed.csv --current data/processed/BBCA.JK_current.csv
    python scripts/generate_drift_report.py --ticker BBCA.JK --split 0.5
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.alerts import AlertSystem
from datetime import datetime


def split_data_by_time(df: pd.DataFrame, split_ratio: float = 0.7):
    """Split data into reference (older) and current (newer) sets."""
    split_index = int(len(df) * split_ratio)
    reference_df = df.iloc[:split_index].copy()
    current_df = df.iloc[split_index:].copy()
    return reference_df, current_df


def generate_report(
    reference_path: Path,
    current_path: Path,
    output_dir: Path,
    target_column: str = "target"
):
    """Generate comprehensive drift report."""
    
    print(f"\n{'='*60}")
    print(f"StockFlowML - Drift Detection Report")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"[DATA] Loading data...")
    print(f"   Reference: {reference_path}")
    print(f"   Current:   {current_path}")
    
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    
    print(f"   OK Reference: {len(reference_data)} rows")
    print(f"   OK Current:   {len(current_data)} rows\n")
    
    # Create drift detector
    detector = DriftDetector(
        reference_data=reference_data,
        current_data=current_data,
        target_column=target_column
    )
    
    # Detect drift
    print(f"[DRIFT] Detecting drift...")
    summary = detector.get_drift_summary()
    
    # Data drift results
    data_drift = summary["data_drift"]
    print(f"\n[RESULTS] Data Drift:")
    print(f"   Dataset Drift: {'ALERT - DETECTED' if data_drift['dataset_drift_detected'] else 'OK - NOT DETECTED'}")
    print(f"   Drift Share:   {data_drift['drift_share']:.2%}")
    print(f"   Drifted Cols:  {data_drift['num_drifted_columns']} / {len(detector.feature_columns)}")
    
    if data_drift["drifted_columns"]:
        print(f"\n   Drifted Features:")
        for col in data_drift["drifted_columns"][:10]:
            print(f"      - {col}")
    
    # Target drift results
    target_drift = summary["target_drift"]
    print(f"\n[TARGET] Target Drift:")
    print(f"   Target Drift:  {'ALERT - DETECTED' if target_drift['target_drift_detected'] else 'OK - NOT DETECTED'}")
    print(f"   Drift Score:   {target_drift.get('target_drift_score', 0):.3f}")
    
    # Generate reports
    print(f"\n[REPORT] Generating reports...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # HTML report
    html_path = output_dir / "drift_report.html"
    detector.generate_html_report(html_path)
    
    # JSON report
    json_path = output_dir / "drift_report.json"
    detector.generate_json_report(json_path)
    
    # Markdown summary
    md_path = output_dir / "drift_report.md"
    generate_markdown_report(summary, data_drift, target_drift, md_path, reference_data, current_data)
    
    # Check thresholds
    print(f"\n[CHECK] Threshold Check:")
    has_drift = detector.has_significant_drift()
    if has_drift:
        print(f"   *** ALERT: Significant drift detected!")
        print(f"   Action: Review drift report and consider retraining model")
    else:
        print(f"   OK: No significant drift - model performance likely stable")
    
    print(f"\n{'='*60}")
    print(f"[DONE] Drift detection complete!")
    print(f"{'='*60}\n")
    
    return has_drift, summary


def generate_markdown_report(
    summary: dict,
    data_drift: dict,
    target_drift: dict,
    output_path: Path,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
):
    """Generate markdown drift report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    alert_status = "ALERT - DRIFT DETECTED" if (data_drift['dataset_drift_detected'] or target_drift['target_drift_detected']) else "OK - NO SIGNIFICANT DRIFT"
    
    content = f"""# StockFlowML - Drift Detection Report

> **Generated**: {timestamp}  
> **Status**: {alert_status}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Reference Size** | {len(reference_data)} rows |
| **Current Size** | {len(current_data)} rows |
| **Features Analyzed** | {len([c for c in reference_data.columns if c not in ['date', 'target']])} |
| **Dataset Drift** | {'YES' if data_drift['dataset_drift_detected'] else 'NO'} |
| **Target Drift** | {'YES' if target_drift['target_drift_detected'] else 'NO'} |

---

## Data Drift Analysis

### Overview

**Drift Detection**: {'**DRIFT DETECTED**' if data_drift['dataset_drift_detected'] else '**NO DRIFT**'}

- **Drift Share**: {data_drift['drift_share']:.2%} of features show drift
- **Drifted Features**: {data_drift['num_drifted_columns']} out of {len([c for c in reference_data.columns if c not in ['date', 'target']])}

### Drifted Features

"""
    
    if data_drift["drifted_columns"]:
        content += "The following features show significant distribution changes:\n\n"
        for i, col in enumerate(data_drift["drifted_columns"], 1):
            content += f"{i}. `{col}`\n"
    else:
        content += "OK - No features show significant drift.\n"
    
    content += f"""

---

## Target Drift Analysis

**Target Drift**: {'**DETECTED**' if target_drift['target_drift_detected'] else '**NOT DETECTED**'}

- **Drift Score**: {target_drift.get('target_drift_score', 0):.3f}
- **Implication**: {'Model may be less accurate on new data distribution' if target_drift['target_drift_detected'] else 'Target distribution stable'}

---

## Recommendations

"""
    
    if data_drift['dataset_drift_detected'] or target_drift['target_drift_detected']:
        content += """
### Action Required

1. **Review Drift Report**: Open `drift_report.html` for detailed visualizations
2. **Investigate Root Cause**: 
   - Market regime change?
   - Data quality issues?
   - Seasonal patterns?
3. **Consider Retraining**: Model may benefit from updated training data
4. **Monitor Performance**: Track model accuracy on new data
5. **Update Alerts**: Configure drift threshold if false positive

### Next Steps

- [ ] Review HTML drift report
- [ ] Analyze drifted feature distributions
- [ ] Retrain model with recent data
- [ ] Update monitoring thresholds if needed
"""
    else:
        content += """
### No Action Needed

- Model is performing on stable data distribution
- Continue monitoring in next cycle
- No immediate retraining required

### Maintenance

- Keep monitoring drift in weekly cycles
- Archive this report for historical tracking
- Update baseline if market conditions change permanently
"""
    
    content += f"""

---

## Technical Details

**Drift Detection Method**: Evidently AI  
**Threshold (Data)**: 50% of features  
**Threshold (Target)**: 0.3 drift score  
**Report Format**: HTML, JSON, Markdown

**Files Generated**:
- `drift_report.html` - Interactive visualization
- `drift_report.json` - Programmatic access
- `drift_report.md` - This summary

---

*Report generated by StockFlowML Drift Detection Pipeline*
"""
    
    output_path.write_text(content)
    print(f"   OK Markdown report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate drift detection report using Evidently AI"
    )
    
    parser.add_argument(
        "--reference",
        type=Path,
        help="Path to reference (baseline) processed CSV"
    )
    parser.add_argument(
        "--current",
        type=Path,
        help="Path to current (new) processed CSV"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="BBCA.JK",
        help="Stock ticker (if using auto-split mode)"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.7,
        help="Train/test split ratio for auto-split (default: 0.7)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="target",
        help="Name of target column"
    )
    parser.add_argument(
        "--send-alert",
        action="store_true",
        help="Send Discord alert if drift detected (requires DISCORD_WEBHOOK_URL env var)"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.reference and args.current:
        reference_path = args.reference
        current_path = args.current
    else:
        # Auto-split mode
        data_path = Path(f"data/processed/{args.ticker}_processed.csv")
        
        if not data_path.exists():
            print(f"[ERROR] Data file not found: {data_path}")
            print(f"   Run: python -m pipelines.train_pipeline --ticker {args.ticker}")
            sys.exit(1)
        
        print(f"[AUTO-SPLIT] Loading {data_path}")
        df = pd.read_csv(data_path)
        
        reference_df, current_df = split_data_by_time(df, args.split)
        
        # Save temporary files
        temp_dir = Path("data/temp")
        temp_dir.mkdir(exist_ok=True)
        
        reference_path = temp_dir / f"{args.ticker}_reference.csv"
        current_path = temp_dir / f"{args.ticker}_current.csv"
        
        reference_df.to_csv(reference_path, index=False)
        current_df.to_csv(current_path, index=False)
        
        print(f"   Split: {len(reference_df)} reference / {len(current_df)} current")
    
    # Generate report
    has_drift, summary = generate_report(
        reference_path=reference_path,
        current_path=current_path,
        output_dir=args.output_dir,
        target_column=args.target_column
    )
    
    # Send alert if enabled
    if args.send_alert and has_drift:
        alert_system = AlertSystem()
        print("\n[ALERT] Sending Discord notification...")
        alert_system.send_drift_alert(
            ticker=args.ticker if not args.reference else "Stock",
            drift_summary=summary,
            report_path=args.output_dir / "drift_report.html"
        )
    
    # Exit code for CI/CD
    sys.exit(1 if has_drift else 0)


if __name__ == "__main__":
    main()