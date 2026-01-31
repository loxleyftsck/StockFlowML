"""
Model evaluation utilities for StockFlowML.
Provides baseline classifiers, metrics calculation, and validation checks
for ensuring model quality and temporal integrity.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.dummy import DummyClassifier
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BaselineClassifiers:
    """Baseline classifiers for comparison."""
    
    @staticmethod
    def random_classifier(X_train, y_train, X_test) -> np.ndarray:
        """
        Random classifier (stratified by class distribution).
        
        Args:
            X_train, y_train: Training data (for fitting class distribution)
            X_test: Test features
            
        Returns:
            Predictions
        """
        clf = DummyClassifier(strategy='stratified', random_state=42)
        clf.fit(X_train, y_train)
        return clf.predict(X_test)
    
    @staticmethod
    def majority_class_classifier(X_train, y_train, X_test) -> np.ndarray:
        """
        Majority class classifier (always predicts most common class).
        
        Args:
            X_train, y_train: Training data (for finding majority class)
            X_test: Test features
            
        Returns:
            Predictions
        """
        clf = DummyClassifier(strategy='most_frequent', random_state=42)
        clf.fit(X_train, y_train)
        return clf.predict(X_test)


class ModelEvaluator:
    """Comprehensive model evaluation and validation."""
    
    def __init__(self, model, X_train, X_test, y_train, y_test):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model with predict() method
            X_train, X_test: Features
            y_train, y_test: Labels
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.metrics = {}
        self.baseline_metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, prefix="") -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            f'{prefix}accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}precision': precision_score(y_true, y_pred, zero_division=0),
            f'{prefix}recall': recall_score(y_true, y_pred, zero_division=0),
            f'{prefix}f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics[f'{prefix}confusion_matrix'] = cm
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        metrics[f'{prefix}class_distribution'] = dict(zip(unique, counts))
        
        return metrics
    
    def evaluate_model(self) -> Dict:
        """
        Evaluate model on train and test sets.
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model performance...")
        
        # Get predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(self.y_train, y_train_pred, prefix='train_')
        test_metrics = self.calculate_metrics(self.y_test, y_test_pred, prefix='test_')
        
        self.metrics = {**train_metrics, **test_metrics}
        
        logger.info(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        
        return self.metrics
    
    def evaluate_baselines(self) -> Dict:
        """
        Evaluate baseline classifiers.
        
        Returns:
            Dictionary of baseline metrics
        """
        logger.info("Evaluating baseline classifiers...")
        
        # Random classifier
        random_preds = BaselineClassifiers.random_classifier(
            self.X_train, self.y_train, self.X_test
        )
        random_metrics = self.calculate_metrics(self.y_test, random_preds, prefix='baseline_random_')
        
        # Majority class classifier
        majority_preds = BaselineClassifiers.majority_class_classifier(
            self.X_train, self.y_train, self.X_test
        )
        majority_metrics = self.calculate_metrics(self.y_test, majority_preds, prefix='baseline_majority_')
        
        self.baseline_metrics = {**random_metrics, **majority_metrics}
        
        logger.info(f"Random Baseline Accuracy: {random_metrics['baseline_random_accuracy']:.4f}")
        logger.info(f"Majority Baseline Accuracy: {majority_metrics['baseline_majority_accuracy']:.4f}")
        
        return self.baseline_metrics
    
    def compare_to_baselines(self) -> Dict:
        """
        Compare model performance to baselines.
        
        Returns:
            Dictionary with comparison results
        """
        if not self.metrics or not self.baseline_metrics:
            raise ValueError("Run evaluate_model() and evaluate_baselines() first")
        
        test_acc = self.metrics['test_accuracy']
        random_acc = self.baseline_metrics['baseline_random_accuracy']
        majority_acc = self.baseline_metrics['baseline_majority_accuracy']
        
        comparison = {
            'beats_random': test_acc > random_acc,
            'beats_majority': test_acc > majority_acc,
            'improvement_over_random': test_acc - random_acc,
            'improvement_over_majority': test_acc - majority_acc,
            'suspicious_high_accuracy': test_acc > 0.70,  # Flag if > 70%
        }
        
        logger.info(f"Beats random: {comparison['beats_random']}")
        logger.info(f"Beats majority: {comparison['beats_majority']}")
        
        if comparison['suspicious_high_accuracy']:
            logger.warning(f"⚠️  SUSPICIOUS: Test accuracy {test_acc:.2%} exceeds 70% threshold")
            logger.warning("   This may indicate data leakage or overfitting")
        
        return comparison
    
    def check_class_imbalance(self) -> Dict:
        """
        Analyze class distribution and imbalance.
        
        Returns:
            Dictionary with class balance analysis
        """
        train_dist = self.metrics['train_class_distribution']
        test_dist = self.metrics['test_class_distribution']
        
        # Get predictions
        y_test_pred = self.model.predict(self.X_test)
        pred_dist = dict(zip(*np.unique(y_test_pred, return_counts=True)))
        
        imbalance_info = {
            'train_class_0_pct': train_dist.get(0, 0) / len(self.y_train),
            'train_class_1_pct': train_dist.get(1, 0) / len(self.y_train),
            'test_class_0_pct': test_dist.get(0, 0) / len(self.y_test),
            'test_class_1_pct': test_dist.get(1, 0) / len(self.y_test),
            'pred_class_0_pct': pred_dist.get(0, 0) / len(y_test_pred),
            'pred_class_1_pct': pred_dist.get(1, 0) / len(y_test_pred),
            'trivial_prediction': max(pred_dist.get(0, 0), pred_dist.get(1, 0)) / len(y_test_pred) > 0.95
        }
        
        if imbalance_info['trivial_prediction']:
            logger.warning("⚠️  Model is trivially predicting one class (>95%)")
        
        return imbalance_info
    
    def check_reproducibility(self, model_class, X_train, y_train, X_test, n_runs=3, tolerance=0.01):
        """
        Check model reproducibility across multiple runs.
        
        Args:
            model_class: Class of model to instantiate
            X_train, y_train: Training data
            X_test: Test features
            n_runs: Number of runs to test
            tolerance: Acceptable variance in metrics
            
        Returns:
            Dictionary with reproducibility results
        """
        logger.info(f"Checking reproducibility over {n_runs} runs...")
        
        accuracies = []
        all_predictions = []
        
        for i in range(n_runs):
            # Create fresh model instance
            model_instance = model_class
            model_instance.train(X_train, y_train)
            
            # Get predictions
            preds = model_instance.predict(X_test)
            all_predictions.append(preds)
            
            # Calculate accuracy
            acc = accuracy_score(self.y_test, preds)
            accuracies.append(acc)
        
        # Check consistency
        accuracy_std = np.std(accuracies)
        accuracy_range = max(accuracies) - min(accuracies)
        
        # Check if all predictions are identical
        predictions_identical = all(
            np.array_equal(all_predictions[0], pred) for pred in all_predictions[1:]
        )
        
        reproducibility = {
            'accuracies': accuracies,
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': accuracy_std,
            'accuracy_range': accuracy_range,
            'within_tolerance': accuracy_range <= tolerance,
            'predictions_identical': predictions_identical,
            'deterministic': predictions_identical and accuracy_range == 0
        }
        
        if not reproducibility['deterministic']:
            logger.warning(f"⚠️  Model is NON-DETERMINISTIC")
            logger.warning(f"   Accuracy range: {accuracy_range:.6f}")
        
        return reproducibility
    
    def generate_report_dict(self) -> Dict:
        """
        Generate comprehensive evaluation report as dictionary.
        
        Returns:
            Dictionary containing all evaluation results
        """
        comparison = self.compare_to_baselines()
        imbalance = self.check_class_imbalance()
        
        report = {
            'model_metrics': self.metrics,
            'baseline_metrics': self.baseline_metrics,
            'comparison': comparison,
            'class_imbalance': imbalance,
            'dataset_info': {
                'train_size': len(self.y_train),
                'test_size': len(self.y_test),
                'n_features': self.X_train.shape[1],
            }
        }
        
        return report


def detect_leakage_in_features(feature_df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Detect potential data leakage in features.
    
    Checks:
    - Target column in features
    - Features with perfect correlation to target
    
    Args:
        feature_df: DataFrame with features and target
        target_col: Name of target column
        
    Returns:
        List of warnings/issues found
    """
    issues = []
    
    # Get feature columns (exclude target and Date)
    feature_cols = [col for col in feature_df.columns if col not in [target_col, 'Date']]
    
    # Check if target is accidentally in features
    if target_col in feature_cols:
        issues.append(f"CRITICAL: Target column '{target_col}' found in features!")
    
    # Check for perfect correlations
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            corr = feature_df[[col, target_col]].corr().iloc[0, 1]
            if abs(corr) > 0.99:
                issues.append(f"SUSPICIOUS: Feature '{col}' has correlation {corr:.4f} with target")
    
    return issues


def validate_temporal_split(train_df: pd.DataFrame, test_df: pd.DataFrame, date_col: str = 'Date') -> Dict:
    """
    Validate that train/test split respects temporal ordering.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        date_col: Name of date column
        
    Returns:
        Dictionary with validation results
    """
    if date_col not in train_df.columns or date_col not in test_df.columns:
        return {'valid': False, 'error': f"Date column '{date_col}' not found"}
    
    train_max_date = train_df[date_col].max()
    test_min_date = test_df[date_col].min()
    
    valid = train_max_date < test_min_date
    
    result = {
        'valid': valid,
        'train_max_date': str(train_max_date),
        'test_min_date': str(test_min_date),
        'gap_days': (test_min_date - train_max_date).days if valid else None
    }
    
    if not valid:
        result['error'] = "Train data contains dates >= test data (temporal leakage!)"
    
    return result
