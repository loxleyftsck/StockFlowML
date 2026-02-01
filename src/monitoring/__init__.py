"""
Monitoring Module - Level 2

This module provides production-grade monitoring capabilities including:
- Data drift detection (Evidently AI)
- Target drift detection
- Model performance tracking
- Alert system (Discord/Email)
"""

from .drift_detector import DriftDetector, load_and_compare

__all__ = [
    "DriftDetector",
    "load_and_compare",
]
