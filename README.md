# StockFlowML

> **Production-ready MLOps Pipeline** for Stock Trend Prediction using Indonesian Stock Exchange data

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

StockFlowML is an end-to-end **MLOps pipeline** that demonstrates industry-standard practices for machine learning in production. The system predicts next-day stock price trends (up/down) using historical OHLCV data and technical indicators.

**Key Principles:**
- ✅ **Reproducibility**: Every experiment is tracked and reproducible via DVC
- ✅ **Automation**: Automated weekly retraining via GitHub Actions
- ✅ **Clean Architecture**: Modular, testable, maintainable code
- ✅ **Production-Ready**: Built for scalability and monitoring

## 🏗️ Architecture

### MLOps Workflow

![StockFlowML MLOps Workflow](docs/images/mlops_workflow.png)

*Complete MLOps workflow with Git branching strategy, CI/CD pipeline, and automated monitoring*

### Development Workflow

- **main branch**: Production-ready models with validated performance
- **development branch**: Experimentation and feature updates  
- **CI Quality Gate**: Automated checks before merge (tests, metrics thresholds, drift detection)

**Pipeline Stages:**
1. **Data Ingestion**: Download OHLCV data from Yahoo Finance (BBCA.JK default)
2. **Feature Engineering**: Create returns, rolling averages, and volatility features
3. **Model Training**: Train Logistic Regression baseline
4. **Evaluation**: Calculate accuracy, precision, recall, F1-score
5. **Reporting**: Generate markdown metrics report

**MLOps Components:**
- **DVC**: Version control for data and models
- **GitHub Actions**: Weekly automated retraining + CI validation
- **Monitoring**: Drift detection and performance tracking

## 📊 Features

### Level 1 (Implemented) ✅
- [x] Automated data fetching from Yahoo Finance
- [x] **Data Quality Framework**:
  - [x] Comprehensive data validation (schema, financial integrity, temporal checks)
  - [x] Data feasibility assessment (6 production-ready criteria)
  - [x] Automated quality reporting (`data_quality_report.md`, `data_feasibility.md`)
  - [x] CSV fallback for offline/demo use
- [x] Feature engineering with rolling windows
- [x] Binary classification (price up/down prediction)
- [x] Logistic Regression baseline model
- [x] XGBoost support (optional)
- [x] Comprehensive evaluation metrics
- [x] DVC for data/model versioning
- [x] GitHub Actions for weekly retraining
- [x] Markdown-based experiment logging
- [x] Production-grade testing suite

### Level 2 (Implemented) ✅
- [x] **Evidently AI for drift detection**
  - [x] Data drift detection (feature distribution changes)
  - [x] Target drift detection (label distribution changes)
  - [x] HTML/JSON/Markdown drift reports
  - [x] CLI drift report generator
- [x] **Alert System**
  - [x] Discord webhook integration
  - [x] Drift detection alerts
  - [x] Performance degradation alerts
  - [x] Training completion notifications
- [x] **Data quality monitoring** (integrated with Level 1)
- [x] **Threshold-based alerting**

### Level 3 (Scaffold) 🔮
- [ ] Feast feature store
- [ ] FastAPI prediction serving
- [ ] Real-time inference
- [ ] Docker deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- Virtualenv (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/loxleyftsck/StockFlowML.git
cd StockFlowML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init
```

### Running the Pipeline

**Full pipeline (recommended):**
```bash
python -m pipelines.train_pipeline
```

**Step-by-step:**
```bash
# 1. Download data
python src/data/data_loader.py

# 2. Feature engineering
python src/features/feature_engineering.py

# 3. Train model
python src/models/train.py

# 4. Evaluate model
python src/evaluation/evaluate.py
```

**With custom ticker:**
```bash
python -m pipelines.train_pipeline --ticker TLKM.JK --model logistic
```

### View Results

After training, check:
- **Model Metrics**: `reports/metrics.md` (training performance)
- **Data Quality Report**: `reports/data_quality_report.md` (validation results)
- **Data Feasibility Report**: `reports/data_feasibility.md` (production readiness)
- **Trained Model**: `models/logistic_model.pkl`
- **Processed Data**: `data/processed/BBCA.JK_processed.csv`

### Level 2: Monitoring & Drift Detection

**Detect data drift:**
```bash
# Auto-split mode (uses 70% as reference, 30% as current)
python scripts/generate_drift_report.py --ticker BBCA.JK --split 0.7

# Explicit mode (compare two datasets)
python scripts/generate_drift_report.py \
  --reference data/processed/BBCA.JK_baseline.csv \
  --current data/processed/BBCA.JK_today.csv
```

**View drift reports:**
- **Interactive HTML**: `reports/drift_report.html` (Evidently AI visualization)
- **Programmatic JSON**: `reports/drift_report.json` (for automation)
- **Executive Summary**: `reports/drift_report.md` (markdown summary)

**Send Discord alerts** (requires `DISCORD_WEBHOOK_URL` env var):
```bash
# Set webhook URL
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_HERE"

# Generate report with alerts
python scripts/generate_drift_report.py --ticker BBCA.JK --send-alert
```

**Test alert system:**
```bash
python -m src.monitoring.alerts
```

**What drift detection monitors:**
- Feature distribution changes (data drift)
- Target label distribution changes (target drift)
- Threshold-based alerting (50% feature drift or 0.3 target drift score)
- Automatic recommendations for retraining


## 📁 Project Structure

```plaintext
StockFlowML/
├── .github/
│   └── workflows/
│       └── retrain.yml          # Weekly automated retraining
├── data/
│   ├── raw/                     # Downloaded stock data (DVC tracked)
│   ├── processed/               # Engineered features (DVC tracked)
│   └── fallback/                # CSV snapshots for offline/demo use
├── models/                      # Trained models (DVC tracked)
├── reports/
│   ├── metrics.md               # Training performance metrics
│   ├── data_quality_report.md   # Data validation results
│   ├── data_feasibility.md      # Production readiness assessment
│   ├── drift_report.html        # Interactive drift visualization (Level 2)
│   ├── drift_report.json        # Drift metrics for automation (Level 2)
│   └── drift_report.md          # Drift summary report (Level 2)
├── scripts/
│   ├── generate_data_quality_report.py    # Data validation report
│   ├── generate_feasibility_report.py     # Feasibility assessment
│   ├── generate_drift_report.py           # Drift detection report (Level 2)
│   ├── generate_fallback_data.py          # Create CSV snapshots
│   └── generate_synthetic_data.py         # Demo data generator
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Yahoo Finance data download
│   │   ├── data_validation.py   # Data contract validation
│   │   └── data_feasibility.py  # Production readiness checks
│   ├── features/
│   │   └── feature_engineering.py  # Rolling windows, returns
│   ├── models/
│   │   └── train.py             # Logistic Regression & XGBoost
│   ├── evaluation/
│   │   └── evaluate.py          # Metrics calculation & reporting
│   ├── monitoring/              # Level 2: Monitoring & Drift Detection
│   │   ├── drift_detector.py    # Evidently AI drift detection
│   │   ├── alerts.py            # Discord/Email alert system
│   │   └── __init__.py          # Monitoring module exports
│   ├── api/                     # FastAPI serving (Level 3)
│   └── utils/
│       └── config.py            # Centralized configuration
├── pipelines/
│   └── train_pipeline.py        # End-to-end orchestration
├── tests/
│   ├── test_pipeline.py         # Integration tests
│   ├── test_data_validation.py  # Data contract tests
│   └── test_data_feasibility.py # Feasibility criteria tests
├── dvc.yaml                     # DVC pipeline definition
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
└── README.md                    # This file
```

---

## 🔒 Data Contract & Quality

> **Production Rule**: No model training occurs unless data passes all validation checks.

### Why Data Quality Matters in Finance

Financial time-series data requires rigorous validation because:
- **Market integrity**: Invalid OHLC relationships indicate data corruption
- **Model reliability**: Garbage in = garbage out - invalid data destroys model performance
- **Regulatory compliance**: Financial systems require auditable data provenance
- **Risk management**: Silent data errors can lead to catastrophic trading decisions

### Data Contract Specification

All data (Yahoo Finance or CSV snapshots) MUST conform to this contract:

#### 1. Schema Requirements
- **Columns**: Date, Open, High, Low, Close, Volume (exact names)
- **Types**:
  - Date: datetime64
  - OHLC: float64
  - Volume: int64

#### 2. Time Properties
- **Frequency**: Daily (one row per trading day)
- **Trading days**: Weekdays only (no weekends)
- **Sorted**: Ascending chronological order
- **Unique**: No duplicate timestamps

#### 3. Financial Validity
- **Positive prices**: Open, High, Low, Close > 0
- **Non-negative volume**: Volume >= 0
- **OHLC relationships**:
  - Low <= Open <= High
  - Low <= Close <= High
  - High >= Low

#### 4. Data Integrity
- **No missing values**: OHLCV columns must be complete
- **No forward-filling**: Missing days are dropped (not filled)
- **No future leakage**: Target generation validated

### Validation Process

```python
# Data is automatically validated during loading
from src.data.data_loader import StockDataLoader
from src.data.data_validation import validate_dataframe

loader = StockDataLoader("BBCA.JK")
df = loader.download_data()
df = loader.clean_data(validate=True)  # Explicit validation

# Check validation stats
print(loader.validation_stats)
```

**Validation Checks**:
1. Schema validation (columns, types)
2. Time properties (sorted, no duplicates)
3. Financial integrity (OHLC relationships, positive prices)
4. Missing data detection (no NaNs in OHLCV)

### Data Quality Report

Generate data quality report:

```bash
python scripts/generate_data_quality_report.py --ticker BBCA.JK
```

View report at: `reports/data_quality_report.md`

### Validation Failures

All validation failures raise `DataValidationError` with actionable messages:

```
DataValidationError: Financial integrity validation failed for BBCA.JK:
Removed 15 invalid rows (15.0%). Threshold: 10%.
Invalid row breakdown: {'high_less_than_low': 8, 'negative_prices': 7}.
Action: Investigate data source quality.
```

**No silent fixes.** All data cleaning is logged and auditable.

### Data Feasibility Assessment

Beyond basic validation, StockFlowML includes **production readiness checks**:

```bash
python scripts/generate_feasibility_report.py --ticker BBCA.JK
```

**6 Production-Ready Criteria**:
1. ✅ **Minimum Samples**: >= 100 trading days for statistical validity
2. ✅ **Data Completeness**: >= 90% of OHLCV data present
3. ✅ **Temporal Continuity**: No gaps > 10 trading days
4. ✅ **Outlier Detection**: < 1% extreme outliers (Z-score > 4.0)
5. ✅ **Label Balance**: Minority class between 20-80%
6. ✅ **Look-Ahead Bias**: No features with future information

View report at: `reports/data_feasibility.md`

## 🛠️ Utility Scripts

StockFlowML includes production-ready scripts for data management and reporting:

### Data Quality & Validation

```bash
# Generate comprehensive data quality report
python scripts/generate_data_quality_report.py --ticker BBCA.JK

# Generate production feasibility assessment
python scripts/generate_feasibility_report.py --ticker DEMO

# Validate real Yahoo Finance data (detailed analysis)
python scripts/generate_model_validation_report.py
```

### Data Management

```bash
# Create CSV snapshot for offline use
python scripts/generate_fallback_data.py --ticker BBCA.JK --days 180

# Generate synthetic OHLCV data for demos
python scripts/generate_synthetic_data.py --ticker DEMO --days 500

# Check snapshot freshness
python scripts/check_snapshot.py
```

**All scripts support**:
- Multiple ticker symbols
- Configurable parameters
- Markdown report generation
- Error handling with clear messages

---

## 🔬 Model Details

### Input Features

**Price-based:**
- Daily returns: `(Close[t] - Close[t-1]) / Close[t-1]`

**Rolling Statistics:**
- Moving averages (5, 10, 20 days)
- Rolling volatility (std of returns, 5/10/20 days)

**Raw OHLCV:**
- Open, High, Low, Close, Volume

### Target Variable

**Binary Classification:**
- `target = 1` if `Close[t+1] > Close[t]` (price goes UP tomorrow)
- `target = 0` if `Close[t+1] <= Close[t]` (price goes DOWN tomorrow)

### Models

1. **Logistic Regression** (Default)
   - Baseline linear classifier
   - Fast training, interpretable
   - Class-balanced for imbalanced data

2. **XGBoost** (Optional)
   - Gradient boosting ensemble
   - Better performance, longer training
   - Use: `--model xgboost`

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: When predicting UP, how often correct?
- **Recall**: Of all UP days, how many caught?
- **F1-Score**: Harmonic mean of precision/recall
- **Confusion Matrix**: Detailed breakdown

## 🔄 CI/CD Pipeline

### Automated Retraining

**Schedule**: Every Friday at 16:00 WIB (after market close)

**GitHub Actions Workflow:**
1. Download latest stock data
2. Run feature engineering
3. Train model
4. Evaluate performance
5. Update `reports/metrics.md`
6. Auto-commit results

**Manual Trigger:**
```bash
# Via GitHub UI: Actions → Weekly Model Retraining → Run workflow
```

### DVC Workflow

```bash
# Reproduce entire pipeline
dvc repro

# Pull data/models from remote
dvc pull

# Push updated artifacts
dvc push
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 📈 Performance Expectations

**Baseline Performance (BBCA.JK, 5 years):**
- Training Accuracy: ~52-55%
- Test Accuracy: ~50-53%

> **Note**: Stock prediction is inherently difficult. Accuracy >50% is better than random. Focus is on MLOps practices, not achieving high accuracy.

## 🛠️ Development

### Adding New Features

1. Edit `src/features/feature_engineering.py`
2. Add new feature calculation methods
3. Update `create_all_features()`
4. Re-run pipeline

### Using Different Stocks

```bash
# Indonesian stocks (IDX)
python -m pipelines.train_pipeline --ticker TLKM.JK  # Telkom
python -m pipelines.train_pipeline --ticker BMRI.JK  # Bank Mandiri

# US stocks
python -m pipelines.train_pipeline --ticker AAPL  # Apple
```

### Hyperparameter Tuning

Edit model parameters in `src/models/train.py`:

```python
# Logistic Regression
LogisticRegression(
    C=1.0,  # Regularization strength
    max_iter=1000,
    solver='lbfgs'
)

# XGBoost
XGBClassifier(
    n_estimators=100,  # Number of trees
    max_depth=5,       # Tree depth
    learning_rate=0.1
)
```

## 📚 MLOps Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| **Data Versioning** | DVC tracks raw data, processed features, models |
| **Reproducibility** | `dvc.yaml` defines exact pipeline stages |
| **Automation** | GitHub Actions for scheduled retraining |
| **Experiment Tracking** | Markdown metrics logged with each run |
| **Clean Architecture** | Modular separation (data/features/models/eval) |
| **Configuration Management** | Centralized `config.py` |
| **Temporal Validation** | Train/test split preserves time order |

## 🗺️ Roadmap

### Level 2: Monitoring (Next)
- [ ] Integrate Evidently AI for data drift detection
- [ ] Automated monitoring reports
- [ ] Alert system (Discord webhooks)
- [ ] Deployment gating based on drift score

### Level 3: Production Serving (Future)
- [ ] Feast feature store for low-latency features
- [ ] FastAPI REST API for predictions
- [ ] Redis online feature cache
- [ ] Docker + docker-compose deployment
- [ ] Kubernetes manifests

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data via yfinance
- **DVC.org** for data version control tools
- **Evidently AI** for drift detection framework
- **Feast** for feature store architecture

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ for demonstrating MLOps best practices**

*Last updated: 2026-02-01*
