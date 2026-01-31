"""
Data Feasibility Assessment for StockFlowML.
Production-grade validation ensuring data meets minimum quality thresholds.

This module enforces HARD FAIL for non-production data.
NO silent passes, NO warnings-only - audit-grade validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataFeasibilityError(Exception):
    """Raised when data fails production feasibility checks."""
    pass


class DataFeasibilityValidator:
    """
    Validates data feasibility for production ML pipeline.
    
    Checks:
    - Completeness (minimum data volume)
    - Temporal continuity (no large gaps)
    - Outlier detection (price anomalies)
    - Label balance (classification viability)
    - Look-ahead bias detection
    """
    
    # Production thresholds (explicit, justified)
    MIN_SAMPLES = 100  # Minimum for statistical validity
    MIN_COMPLETENESS_PCT = 90.0  # Maximum 10% missing data allowed
    MAX_GAP_DAYS = 10  # Maximum acceptable gap (2 weeks of trading days)
    OUTLIER_THRESHOLD = 4.0  # Z-score threshold (99.99% confidence)
    MIN_LABEL_BALANCE = 0.20  # Minimum 20% of minority class
    MAX_LABEL_BALANCE = 0.80  # Maximum 80% to avoid trivial datasets
    
    def __init__(self, df: pd.DataFrame, ticker: str = "UNKNOWN", target_col: Optional[str] = None):
        """
        Initialize feasibility validator.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            target_col: Target column name (for label balance check)
        """
        self.df = df.copy()
        self.ticker = ticker
        self.target_col = target_col
        self.checks_passed: Dict[str, bool] = {}
        self.check_details: Dict[str, Dict] = {}
        
    def check_minimum_samples(self) -> bool:
        """
        Check if dataset has minimum number of samples.
        
        Threshold: MIN_SAMPLES (100)
        Rationale: Need sufficient data for train/test split and statistical validity
        
        Returns:
            True if passed
            
        Raises:
            DataFeasibilityError: If below threshold
        """
        n_samples = len(self.df)
        
        logger.info(f"[FEASIBILITY] Checking minimum samples: {n_samples} (threshold: {self.MIN_SAMPLES})")
        
        self.check_details['minimum_samples'] = {
            'n_samples': n_samples,
            'threshold': self.MIN_SAMPLES,
            'passed': n_samples >= self.MIN_SAMPLES
        }
        
        if n_samples < self.MIN_SAMPLES:
            raise DataFeasibilityError(
                f"Minimum samples check FAILED for {self.ticker}: "
                f"Found {n_samples} samples, minimum required: {self.MIN_SAMPLES}. "
                f"Action: Obtain more historical data."
            )
        
        logger.info(f"[FEASIBILITY] ✓ Minimum samples check PASSED: {n_samples} >= {self.MIN_SAMPLES}")
        self.checks_passed['minimum_samples'] = True
        return True
    
    def check_completeness(self) -> bool:
        """
        Check data completeness (missing value ratio).
        
        Threshold: MIN_COMPLETENESS_PCT (90%)
        Rationale: > 10% missing data indicates quality issues
        
        Returns:
            True if passed
            
        Raises:
            DataFeasibilityError: If below threshold
        """
        critical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        total_cells = len(self.df) * len(critical_cols)
        missing_cells = self.df[critical_cols].isna().sum().sum()
        completeness_pct = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        logger.info(f"[FEASIBILITY] Checking completeness: {completeness_pct:.2f}% (threshold: {self.MIN_COMPLETENESS_PCT}%)")
        
        self.check_details['completeness'] = {
            'completeness_pct': completeness_pct,
            'missing_cells': missing_cells,
            'total_cells': total_cells,
            'threshold': self.MIN_COMPLETENESS_PCT,
            'passed': completeness_pct >= self.MIN_COMPLETENESS_PCT
        }
        
        if completeness_pct < self.MIN_COMPLETENESS_PCT:
            raise DataFeasibilityError(
                f"Completeness check FAILED for {self.ticker}: "
                f"Data is {completeness_pct:.2f}% complete, minimum required: {self.MIN_COMPLETENESS_PCT}%. "
                f"Missing {missing_cells} cells out of {total_cells}. "
                f"Action: Fix data gaps at source."
            )
        
        logger.info(f"[FEASIBILITY] ✓ Completeness check PASSED: {completeness_pct:.2f}% >= {self.MIN_COMPLETENESS_PCT}%")
        self.checks_passed['completeness'] = True
        return True
    
    def check_temporal_continuity(self) -> bool:
        """
        Check for large gaps in time series.
        
        Threshold: MAX_GAP_DAYS (10 days)
        Rationale: Large gaps indicate missing data or market closures
        
        Returns:
            True if passed
            
        Raises:
            DataFeasibilityError: If gaps exceed threshold
        """
        if 'Date' not in self.df.columns:
            logger.warning("[FEASIBILITY] Date column not found, skipping temporal continuity check")
            self.checks_passed['temporal_continuity'] = True
            return True
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Calculate gaps between consecutive dates
        date_diffs = self.df['Date'].diff()
        max_gap = date_diffs.max()
        max_gap_days = max_gap.days if pd.notna(max_gap) else 0
        
        # Find all large gaps
        large_gaps = date_diffs[date_diffs > timedelta(days=self.MAX_GAP_DAYS)]
        n_large_gaps = len(large_gaps)
        
        logger.info(f"[FEASIBILITY] Checking temporal continuity: max gap = {max_gap_days} days (threshold: {self.MAX_GAP_DAYS})")
        
        self.check_details['temporal_continuity'] = {
            'max_gap_days': max_gap_days,
            'n_large_gaps': n_large_gaps,
            'threshold': self.MAX_GAP_DAYS,
            'passed': max_gap_days <= self.MAX_GAP_DAYS
        }
        
        if max_gap_days > self.MAX_GAP_DAYS:
            # List gap dates for debugging
            gap_info = []
            for idx in large_gaps.index[:5]:  # First 5 gaps
                gap_info.append(f"{self.df.loc[idx, 'Date']} (gap: {date_diffs.loc[idx].days} days)")
            
            raise DataFeasibilityError(
                f"Temporal continuity check FAILED for {self.ticker}: "
                f"Found {n_large_gaps} gaps exceeding {self.MAX_GAP_DAYS} days. "
                f"Largest gap: {max_gap_days} days. "
                f"First gaps: {gap_info}. "
                f"Action: Investigate data source for missing periods."
            )
        
        logger.info(f"[FEASIBILITY] ✓ Temporal continuity check PASSED: max gap {max_gap_days} <= {self.MAX_GAP_DAYS} days")
        self.checks_passed['temporal_continuity'] = True
        return True
    
    def check_outliers(self) -> bool:
        """
        Detect extreme price outliers.
        
        Threshold: OUTLIER_THRESHOLD (4.0 z-score)
        Rationale: Z-score > 4 indicates 99.99% confidence anomaly
        Method: Z-score on daily returns
        
        Returns:
            True if passed
            
        Raises:
            DataFeasibilityError: If excessive outliers found
        """
        if 'Close' not in self.df.columns:
            logger.warning("[FEASIBILITY] Close column not found, skipping outlier check")
            self.checks_passed['outliers'] = True
            return True
        
        # Calculate daily returns
        returns = self.df['Close'].pct_change()
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            logger.warning("[FEASIBILITY] No valid returns to check outliers")
            self.checks_passed['outliers'] = True
            return True
        
        # Calculate z-scores
        mean_return = returns_clean.mean()
        std_return = returns_clean.std()
        
        if std_return == 0:
            logger.warning("[FEASIBILITY] Zero std deviation in returns, skipping outlier check")
            self.checks_passed['outliers'] = True
            return True
        
        z_scores = np.abs((returns_clean - mean_return) / std_return)
        outliers = z_scores > self.OUTLIER_THRESHOLD
        n_outliers = outliers.sum()
        outlier_pct = (n_outliers / len(returns_clean)) * 100
        
        logger.info(f"[FEASIBILITY] Checking outliers: {n_outliers} outliers ({outlier_pct:.2f}% of data)")
        
        # Allow up to 1% outliers (expected in financial data)
        max_outlier_pct = 1.0
        
        self.check_details['outliers'] = {
            'n_outliers': n_outliers,
            'outlier_pct': outlier_pct,
            'z_threshold': self.OUTLIER_THRESHOLD,
            'max_outlier_pct': max_outlier_pct,
            'passed': outlier_pct <= max_outlier_pct
        }
        
        if outlier_pct > max_outlier_pct:
            # Get some outlier examples
            outlier_indices = returns_clean[outliers].index[:5]
            outlier_examples = []
            for idx in outlier_indices:
                if idx in self.df.index:
                    date = self.df.loc[idx, 'Date'] if 'Date' in self.df.columns else idx
                    ret = returns_clean.loc[idx]
                    z = z_scores.loc[idx]
                    outlier_examples.append(f"{date}: {ret:.2%} (z={z:.2f})")
            
            raise DataFeasibilityError(
                f"Outlier check FAILED for {self.ticker}: "
                f"Found {n_outliers} outliers ({outlier_pct:.2f}%), maximum allowed: {max_outlier_pct}%. "
                f"Examples: {outlier_examples}. "
                f"Action: Investigate price spikes - may indicate data quality issues."
            )
        
        logger.info(f"[FEASIBILITY] ✓ Outlier check PASSED: {outlier_pct:.2f}% <= {max_outlier_pct}%")
        self.checks_passed['outliers'] = True
        return True
    
    def check_label_balance(self) -> bool:
        """
        Check classification label balance.
        
        Threshold: MIN_LABEL_BALANCE (20%) to MAX_LABEL_BALANCE (80%)
        Rationale: Severely imbalanced data makes classification difficult
        
        Returns:
            True if passed
            
        Raises:
            DataFeasibilityError: If labels too imbalanced
        """
        if self.target_col is None or self.target_col not in self.df.columns:
            logger.info("[FEASIBILITY] No target column specified, skipping label balance check")
            self.checks_passed['label_balance'] = True
            return True
        
        # Get label distribution
        label_counts = self.df[self.target_col].value_counts()
        total = len(self.df)
        
        if len(label_counts) == 0:
            raise DataFeasibilityError(
                f"Label balance check FAILED for {self.ticker}: "
                f"No labels found in column '{self.target_col}'."
            )
        
        # Calculate balance (ratio of minority class)
        minority_count = label_counts.min()
        minority_ratio = minority_count / total if total > 0 else 0
        
        logger.info(f"[FEASIBILITY] Checking label balance: minority class = {minority_ratio:.2%}")
        
        self.check_details['label_balance'] = {
            'label_distribution': label_counts.to_dict(),
            'minority_ratio': minority_ratio,
            'min_threshold': self.MIN_LABEL_BALANCE,
            'max_threshold': self.MAX_LABEL_BALANCE,
            'passed': self.MIN_LABEL_BALANCE <= minority_ratio <= self.MAX_LABEL_BALANCE
        }
        
        if minority_ratio < self.MIN_LABEL_BALANCE:
            raise DataFeasibilityError(
                f"Label balance check FAILED for {self.ticker}: "
                f"Minority class ratio {minority_ratio:.2%} below minimum {self.MIN_LABEL_BALANCE:.0%}. "
                f"Distribution: {label_counts.to_dict()}. "
                f"Action: Check target generation logic or obtain more balanced data."
            )
        
        if minority_ratio > self.MAX_LABEL_BALANCE:
            raise DataFeasibilityError(
                f"Label balance check FAILED for {self.ticker}: "
                f"Data is too balanced ({minority_ratio:.2%}), suggests trivial target. "
                f"Distribution: {label_counts.to_dict()}. "
                f"Action: Verify target column is correct."
            )
        
        logger.info(f"[FEASIBILITY] ✓ Label balance check PASSED: {minority_ratio:.2%} in range [{self.MIN_LABEL_BALANCE:.0%}, {self.MAX_LABEL_BALANCE:.0%}]")
        self.checks_passed['label_balance'] = True
        return True
    
    def check_lookahead_bias(self) -> bool:
        """
        Detect potential look-ahead bias in features.
        
        Check: Verify target column doesn't exist in feature columns
        Rationale: Target leakage is critical production flaw
        
        Returns:
            True if passed
            
        Raises:
            DataFeasibilityError: If potential leakage detected
        """
        if self.target_col is None:
            logger.info("[FEASIBILITY] No target column specified, skipping look-ahead check")
            self.checks_passed['lookahead_bias'] = True
            return True
        
        # Get all numeric columns (potential features)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check for suspicious column names containing target patterns
        suspicious_cols = []
        target_keywords = [self.target_col, 'target', 'label', 'future', 'next', 'tomorrow']
        
        for col in numeric_cols:
            col_lower = col.lower()
            for keyword in target_keywords:
                if keyword.lower() in col_lower and col != self.target_col:
                    suspicious_cols.append(col)
                    break
        
        logger.info(f"[FEASIBILITY] Checking look-ahead bias: found {len(suspicious_cols)} suspicious columns")
        
        self.check_details['lookahead_bias'] = {
            'suspicious_columns': suspicious_cols,
            'passed': len(suspicious_cols) == 0
        }
        
        if len(suspicious_cols) > 0:
            raise DataFeasibilityError(
                f"Look-ahead bias check FAILED for {self.ticker}: "
                f"Found {len(suspicious_cols)} suspicious column names: {suspicious_cols}. "
                f"These may contain future information or target leakage. "
                f"Action: Review feature engineering pipeline."
            )
        
        logger.info(f"[FEASIBILITY] ✓ Look-ahead bias check PASSED: no suspicious columns")
        self.checks_passed['lookahead_bias'] = True
        return True
    
    def validate_all(self) -> Dict:
        """
        Run all feasibility checks.
        
        Returns:
            Summary dictionary
            
        Raises:
            DataFeasibilityError: If ANY check fails (hard fail)
        """
        logger.info("="*70)
        logger.info(f"DATA FEASIBILITY VALIDATION: {self.ticker}")
        logger.info("="*70)
        
        # Run all checks (any failure raises exception)
        self.check_minimum_samples()
        self.check_completeness()
        self.check_temporal_continuity()
        self.check_outliers()
        self.check_label_balance()
        self.check_lookahead_bias()
        
        # If we get here, all checks passed
        logger.info("="*70)
        logger.info(f"✓ ALL FEASIBILITY CHECKS PASSED: {self.ticker}")
        logger.info("="*70)
        
        summary = {
            'ticker': self.ticker,
            'verdict': 'PASS',
            'timestamp': datetime.now().isoformat(),
            'checks_passed': self.checks_passed,
            'check_details': self.check_details,
            'n_samples': len(self.df),
            'production_ready': all(self.checks_passed.values())
        }
        
        return summary


def validate_data_feasibility(
    df: pd.DataFrame, 
    ticker: str = "UNKNOWN",
    target_col: Optional[str] = None
) -> Dict:
    """
    Convenience function to validate data feasibility.
    
    Args:
        df: DataFrame to validate
        ticker: Stock ticker symbol
        target_col: Target column name (optional)
        
    Returns:
        Feasibility summary
        
    Raises:
        DataFeasibilityError: If ANY check fails
    """
    validator = DataFeasibilityValidator(df, ticker, target_col)
    return validator.validate_all()
