"""
SHAP-based Model Explainability Module for StockFlowML.

Provides interpretability for stock trend prediction models using SHAP (SHapley Additive exPlanations).
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    SHAP-based explainer for stock trend prediction models.
    
    Supports:
    - Logistic Regression (KernelExplainer)
    - XGBoost (TreeExplainer)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        background_data: Optional[pd.DataFrame] = None,
        n_background_samples: int = 100
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model_path: Path to trained model (.pkl file)
            background_data: Optional background dataset for KernelExplainer
            n_background_samples: Number of background samples to use (default: 100)
        """
        self.model_path = Path(model_path)
        self.n_background_samples = n_background_samples
        
        # Load model artifact
        self.model_artifact = joblib.load(self.model_path)
        self.model = self.model_artifact['model']
        self.scaler = self.model_artifact['scaler']
        self.feature_names = self.model_artifact.get('feature_names', [])
        self.model_type = self.model_artifact.get('model_type', 'unknown')
        
        logger.info(f"Loaded model: {self.model_type}")
        logger.info(f"Features: {self.feature_names}")
        
        # Initialize SHAP explainer
        self.explainer = None
        self.background_data = background_data
        
        if background_data is not None:
            self._initialize_explainer(background_data)
    
    def _initialize_explainer(self, background_data: pd.DataFrame):
        """Initialize appropriate SHAP explainer based on model type."""
        # Sample background data
        if len(background_data) > self.n_background_samples:
            background_sample = shap.sample(background_data, self.n_background_samples)
        else:
            background_sample = background_data
        
        # Scale background data
        background_scaled = self.scaler.transform(background_sample)
        
        try:
            # Try TreeExplainer first (for XGBoost, Random Forest, etc.)
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = 'tree'
            logger.info("Initialized TreeExplainer")
        except Exception as e:
            # Fall back to KernelExplainer (for Logistic Regression, SVM, etc.)
            logger.info(f"TreeExplainer failed: {e}, using KernelExplainer")
            
            # Create prediction function
            def model_predict(X):
                return self.model.predict_proba(X)[:, 1]
            
            self.explainer = shap.KernelExplainer(model_predict, background_scaled)
            self.explainer_type = 'kernel'
            logger.info("Initialized KernelExplainer")
    
    def explain_prediction(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_plots: bool = False
    ) -> Dict:
        """
        Generate SHAP explanation for a single prediction or batch.
        
        Args:
            X: Input features (DataFrame or numpy array)
            return_plots: Whether to generate and return plot objects
        
        Returns:
            Dictionary containing SHAP values and optional plots
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Provide background_data first.")
        
        # Ensure DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
            # For binary classification, TreeExplainer returns values for both classes
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use class 1 (UP)
        else:
            shap_values = self.explainer.shap_values(X_scaled)
        
        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                base_value = self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0]
            else:
                base_value = self.explainer.expected_value
        else:
            base_value = 0.0
        
        # Get actual prediction
        prediction_proba = self.model.predict_proba(X_scaled)[:, 1]
        prediction_class = self.model.predict(X_scaled)
        
        result = {
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'base_value': float(base_value),
            'feature_names': self.feature_names,
            'prediction_probability': prediction_proba.tolist(),
            'prediction_class': prediction_class.tolist(),
            'feature_contributions': {
                fname: float(shap_values[0, i]) if shap_values.ndim > 1 else float(shap_values[i])
                for i, fname in enumerate(self.feature_names)
            }
        }
        
        if return_plots:
            result['plots'] = self._generate_plots(X_scaled, shap_values, base_value)
        
        return result
    
    def _generate_plots(self, X_scaled, shap_values, base_value):
        """Generate SHAP visualization plots."""
        plots = {}
        
        # Waterfall plot (for single prediction)
        if shap_values.ndim == 1 or shap_values.shape[0] == 1:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                shap_val = shap_values[0] if shap_values.ndim > 1 else shap_values
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_val,
                        base_values=base_value,
                        data=X_scaled[0] if X_scaled.ndim > 1 else X_scaled,
                        feature_names=self.feature_names
                    ),
                    show=False
                )
                plots['waterfall'] = fig
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate waterfall plot: {e}")
        
        return plots
    
    def explain_global(
        self,
        X: pd.DataFrame,
        max_display: int = 10
    ) -> Dict:
        """
        Generate global feature importance using SHAP.
        
        Args:
            X: Dataset to explain (typically test set)
            max_display: Maximum number of features to display
        
        Returns:
            Dictionary with global SHAP values and feature importance
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Provide background_data first.")
        
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        # Get SHAP values for dataset
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = self.explainer.shap_values(X_scaled)
        
        # Calculate mean absolute SHAP values (feature importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dictionary
        feature_importance = {
            fname: float(mean_abs_shap[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_display])
        
        return {
            'feature_importance': sorted_importance,
            'shap_values': shap_values.tolist(),
            'num_samples': len(X)
        }
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        plot_type: str= 'bar',
        max_display: int = 10,
        save_path: Optional[Path] = None
    ):
        """
        Generate and optionally save SHAP summary plot.
        
        Args:
            X: Dataset to explain
            plot_type: 'bar' or 'dot'
            max_display: Maximum features to display
            save_path: Optional path to save plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = self.explainer.shap_values(X_scaled)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Summary plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def load_explainer_from_model(model_path:  Union[str, Path]) -> ModelExplainer:
    """
    Convenience function to load explainer from model path.
    
    Args:
        model_path: Path to trained model
    
    Returns:
        ModelExplainer instance
    """
    return ModelExplainer(model_path)
