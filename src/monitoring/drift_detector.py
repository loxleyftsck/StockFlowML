"""
Drift Detection Module using Evidently AI

This module provides data drift and target drift detection capabilities
for monitoring model performance degradation in production.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
import json
from datetime import datetime


class DriftDetector:
    """
    Detects data drift and target drift using Evidently AI.
    
    Compares reference (baseline) data with current (production) data
    to identify distribution changes that may degrade model performance.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = "target",
        data_drift_threshold: float = 0.5,
        target_drift_threshold: float = 0.3
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Baseline dataset (e.g., training data)
            current_data: New dataset to compare against baseline
            target_column: Name of target column
            data_drift_threshold: Drift detection threshold for features (0-1)
            target_drift_threshold: Drift detection threshold for target (0-1)
        """
        self.reference_data = reference_data
        self.current_data = current_data
        self.target_column = target_column
        self.data_drift_threshold = data_drift_threshold
        self.target_drift_threshold = target_drift_threshold
        
        # Separate features and target
        self.feature_columns = [
            col for col in reference_data.columns 
            if col != target_column and col != "date"
        ]
        
    def detect_data_drift(self) -> Dict[str, Any]:
        """
        Detect feature drift using Evidently's DataDriftPreset.
        
        Returns:
            Dict containing drift detection results
        """
        # Create Evidently report for data drift
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=None
        )
        
        # Extract drift metrics
        result_dict = report.as_dict()
        
        # Parse key metrics
        drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift_detected": False,
            "drift_share": 0.0,
            "num_drifted_columns": 0,
            "drifted_columns": [],
        }
        
        try:
            # Navigate the nested structure
            metrics = result_dict.get("metrics", [])
            for metric in metrics:
                if metric.get("metric") == "DatasetDriftMetric":
                    metric_result = metric.get("result", {})
                    drift_summary["dataset_drift_detected"] = metric_result.get("dataset_drift", False)
                    drift_summary["drift_share"] = metric_result.get("drift_share", 0.0)
                    drift_summary["num_drifted_columns"] = metric_result.get("number_of_drifted_columns", 0)
                    
                    # Extract drifted column names
                    drift_by_columns = metric_result.get("drift_by_columns", {})
                    drift_summary["drifted_columns"] = [
                        col for col, info in drift_by_columns.items()
                        if isinstance(info, dict) and info.get("drift_detected", False)
                    ]
                    break
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not parse drift metrics: {e}")
        
        return drift_summary
    
    def detect_target_drift(self) -> Dict[str, Any]:
        """
        Detect target (label) drift using Evidently's TargetDriftPreset.
        
        Returns:
            Dict containing target drift results
        """
        # Create Evidently report for target drift
        report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=None
        )
        
        result_dict = report.as_dict()
        
        # Parse target drift metrics
        target_drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "target_drift_detected": False,
            "target_drift_score": 0.0,
        }
        
        try:
            metrics = result_dict.get("metrics", [])
            for metric in metrics:
                # Look for target-specific drift metric
                metric_name = metric.get("metric", "")
                if "ColumnDriftMetric" in metric_name:
                    metric_result = metric.get("result", {})
                    column_name = metric_result.get("column_name", "")
                    if column_name == self.target_column:
                        target_drift_summary["target_drift_detected"] = metric_result.get("drift_detected", False)
                        target_drift_summary["target_drift_score"] = metric_result.get("drift_score", 0.0)
                        break
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not parse target drift metrics: {e}")
        
        return target_drift_summary
    
    def generate_html_report(self, output_path: Path) -> None:
        """
        Generate comprehensive HTML drift report using Evidently.
        
        Args:
            output_path: Path to save HTML report
        """
        # Create comprehensive report with both data and target drift
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=None
        )
        
        # Save HTML report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))
        print(f"   OK HTML drift report saved to: {output_path}")
    
    def generate_json_report(self, output_path: Path) -> None:
        """
        Generate JSON drift report for programmatic access.
        
        Args:
            output_path: Path to save JSON report
        """
        # Combine all drift metrics
        drift_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "reference_size": len(self.reference_data),
                "current_size": len(self.current_data),
                "feature_count": len(self.feature_columns),
            },
            "data_drift": self.detect_data_drift(),
            "target_drift": self.detect_target_drift(),
        }
        
        # Save JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(drift_report, f, indent=2)
        
        print(f"   OK JSON drift report saved to: {output_path}")
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get complete drift summary (convenience method).
        
        Returns:
            Dict with data drift and target drift results
        """
        return {
            "data_drift": self.detect_data_drift(),
            "target_drift": self.detect_target_drift(),
        }
    
    def has_significant_drift(self) -> bool:
        """
        Check if significant drift detected based on thresholds.
        
        Returns:
            True if drift exceeds thresholds, False otherwise
        """
        data_drift = self.detect_data_drift()
        target_drift = self.detect_target_drift()
        
        data_drift_significant = data_drift["drift_share"] > self.data_drift_threshold
        target_drift_significant = target_drift.get("target_drift_score", 0) > self.target_drift_threshold
        
        return data_drift_significant or target_drift_significant


def load_and_compare(
    reference_path: Path,
    current_path: Path,
    target_column: str = "target"
) -> DriftDetector:
    """
    Convenience function to load data and create drift detector.
    
    Args:
        reference_path: Path to reference (baseline) CSV
        current_path: Path to current (new) CSV
        target_column: Name of target column
        
    Returns:
        DriftDetector instance
    """
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    
    return DriftDetector(
        reference_data=reference_data,
        current_data=current_data,
        target_column=target_column
    )


if __name__ == "__main__":
    # Example usage
    print("Drift Detector - Example Usage")
    print("-" * 50)
    
    # This is a demo - in production, use actual data paths
    print("Usage:")
    print("  from src.monitoring.drift_detector import DriftDetector")
    print("  detector = DriftDetector(reference_df, current_df)")
    print("  summary = detector.get_drift_summary()")
    print("  detector.generate_html_report(Path('reports/drift_report.html'))")
