"""
Generate comprehensive SHAP explainability report for trained models.

Usage:
    python scripts/generate_explainability_report.py --model models/logistic_model.pkl
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.explainability.shap_explainer import ModelExplainer
from src.utils.config import Config


def generate_explainability_report(
    model_path: Path,
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_background: int = 100,
    n_explain: int = 50
):
    """
    Generate comprehensive SHAP explainability report.
    
    Args:
        model_path: Path to trained model
        data_path: Optional path to processed data (default: use config)
        output_dir: Output directory for reports (default: reports/explainability/)
        n_background: Number of background samples for SHAP
        n_explain: Number of samples to explain
    """
    # Setup output directory
    if output_dir is None:
        output_dir = project_root / "reports" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generating SHAP Explainability Report")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Load processed data
    if data_path is None:
        data_path = project_root / "data" / "processed" / f"{Config.DEFAULT_TICKER}_processed.csv"
        # Try DEMO if default doesn't exist
        if not data_path.exists():
            data_path = project_root / "data" / "processed" / "DEMO_processed.csv"
    
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        print("Run training pipeline first: python -m pipelines.train_pipeline")
        return False
    
    print(f"📂 Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Load model first to get correct feature names
    print(f"\n🔧 Loading model to get feature names...")
    import joblib
    model_artifact = joblib.load(model_path)
    feature_cols = model_artifact.get('feature_names', [])
    
    if not feature_cols:
        # Fallback to standard features
        feature_cols = ['returns', 'ma_5', 'ma_10', 'ma_20', 'volatility_5', 'volatility_10', 'volatility_20']
        print(f"⚠️  Model has no feature_names, using default: {feature_cols}")
    else:
        print(f"✅ Model features: {feature_cols}")
    
    # Check if we have all features
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        print(f"❌ Missing features in data: {missing_features}")
        return False
    
    X = df[feature_cols].dropna()
    
    if len(X) == 0:
        print(f"❌ No valid samples after removing NaN")
        return False
    
    print(f"✅ Loaded {len(X)} samples")
    
    # Initialize explainer
    print(f"\n🔧 Initializing SHAP explainer...")
    background_data = X.sample(min(n_background, len(X)), random_state=42)
    explainer = ModelExplainer(
        model_path=model_path,
        background_data=background_data,
        n_background_samples=n_background
    )
    
    print(f"✅ Explainer initialized ({explainer.explainer_type})")
    
    # 1. Global Feature Importance
    print(f"\n📈 Computing global feature importance...")
    explain_samples = X.sample(min(n_explain, len(X)), random_state=42)
    global_explanation = explainer.explain_global(explain_samples, max_display=len(feature_cols))
    
    print("\n🌟 Top Feature Importance:")
    for i, (feature, importance) in enumerate(global_explanation['feature_importance'].items(), 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # 2. Generate Summary Plot
    print(f"\n📊 Generating summary plot...")
    summary_plot_path = output_dir / "summary_plot.png"
    explainer.plot_summary(
        explain_samples,
        plot_type='bar',
        max_display=len(feature_cols),
        save_path=summary_plot_path
    )
    print(f"✅ Saved: {summary_plot_path}")
    
    # 3. Individual Prediction Explanations (first 3 samples)
    print(f"\n🔍 Generating individual prediction explanations...")
    sample_explanations = []
    for i in range(min(3, len(explain_samples))):
        sample = explain_samples.iloc[[i]]
        explanation = explainer.explain_prediction(sample)
        sample_explanations.append({
            'index': i,
            'prediction': explanation['prediction_class'][0],
            'probability': explanation['prediction_probability'][0],
            'contributions': explanation['feature_contributions']
        })
        print(f"  Sample {i+1}: Prediction={explanation['prediction_class'][0]} ({explanation['prediction_probability'][0]:.2%})")
    
    # 4. Generate Markdown Report
    print(f"\n📝 Generating markdown report...")
    report_path = output_dir / "explainability_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Stock Trend Prediction Model - Explainability Report\n\n")
        f.write(f"**Model:** `{model_path.name}`\n\n")
        f.write(f"**Model Type:** {explainer.model_type}\n\n")
        f.write(f"**Explainer Type:** SHAP {explainer.explainer_type.capitalize()}Explainer\n\n")
        f.write(f"**Samples Analyzed:** {len(explain_samples)}\n\n")
        f.write("---\n\n")
        
        # Global Importance
        f.write("## 📊 Global Feature Importance\n\n")
        f.write("Mean absolute SHAP values (higher = more important):\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for i, (feature, importance) in enumerate(global_explanation['feature_importance'].items(), 1):
            f.write(f"| {i} | `{feature}` | {importance:.4f} |\n")
        
        f.write(f"\n![Feature Importance](summary_plot.png)\n\n")
        f.write("*Bar plot showing mean absolute SHAP values for each feature.*\n\n")
        
        # Interpretation
        f.write("## 💡 Interpretation\n\n")
        top_feature = list(global_explanation['feature_importance'].keys())[0]
        f.write(f"The most important feature for predictions is **`{top_feature}`**, ")
        f.write("indicating it has the strongest influence on whether the model predicts UP or DOWN.\n\n")
        
        # Individual Examples
        f.write("## 🔍 Individual Prediction Examples\n\n")
        for ex in sample_explanations:
            pred_label = "📈 UP" if ex['prediction'] == 1 else "📉 DOWN"
            f.write(f"### Sample {ex['index'] + 1}: {pred_label} ({ex['probability']:.2%} confidence)\n\n")
            f.write("Feature contributions (positive = pushes toward UP, negative = pushes toward DOWN):\n\n")
            f.write("| Feature | Contribution |\n")
            f.write("|---------|-------------|\n")
            
            # Sort by absolute contribution
            sorted_contrib = sorted(ex['contributions'].items(), key=lambda x: abs(x[1]), reverse=True)
            for feature, contrib in sorted_contrib:
                emoji = "🔼" if contrib > 0 else "🔽"
                f.write(f"| `{feature}` | {emoji} {contrib:+.4f} |\n")
            f.write("\n")
        
        # How to Use
        f.write("## 📖 How to Use SHAP Values\n\n")
        f.write("1. **Global Importance**: Higher values mean the feature has more impact on predictions\n")
        f.write("2. **Individual Contributions**: Positive values push toward UP prediction, negative toward DOWN\n")
        f.write("3. **SHAP values add up**: Base value + sum of contributions = final prediction\n\n")
        
        f.write("## ⚙️ Technical Details\n\n")
        f.write(f"- **Model**: {explainer.model_type}\n")
        f.write(f"- **Features**: {len(explainer.feature_names)}\n")
        f.write(f"- **Background Samples**: {n_background}\n")
        f.write(f"- **Explained Samples**: {len(explain_samples)}\n")
        f.write(f"- **SHAP Method**: {explainer.explainer_type.capitalize()}Explainer\n\n")
    
    print(f"✅ Saved: {report_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Explainability report generated successfully!")
    print(f"📂 Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explainability report")
    parser.add_argument(
        "--model",
        type=str,
        default="models/logistic_model.pkl",
        help="Path to trained model (default: models/logistic_model.pkl)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to processed data (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: reports/explainability/)"
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=100,
        help="Number of background samples for SHAP (default: 100)"
    )
    parser.add_argument(
        "--n-explain",
        type=int,
        default=50,
        help="Number of samples to explain (default: 50)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    data_path = Path(args.data) if args.data else None
    output_dir = Path(args.output) if args.output else None
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Train a model first: python -m pipelines.train_pipeline")
        sys.exit(1)
    
    success = generate_explainability_report(
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir,
        n_background=args.n_background,
        n_explain=args.n_explain
    )
    
    sys.exit(0 if success else 1)

