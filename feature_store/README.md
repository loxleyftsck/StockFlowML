# Feast Feature Store - StockFlowML

Production-ready feature store implementation using Feast for low-latency feature serving.

## 🎯 Overview

The Feast feature store enables:
- **Real-time feature retrieval** for online predictions
- **Offline feature storage** for model training
- **Feature versioning** and consistency
- **Low-latency serving** (< 10ms feature retrieval)

## 📂 Structure

```
feature_store/
├── feature_repo/
│   ├── feature_store.yaml    # Feast configuration
│   ├── features.py            # Feature definitions
│   └── __init__.py
├── materialize_features.py    # Materialization script
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Prepare Feature Data

Convert processed CSV to Parquet format:

```bash
python feature_store/materialize_features.py --ticker BBCA.JK --prepare-only
```

This creates `data/processed/feast_features.parquet`.

### 2. Initialize Feast

```bash
cd feature_store/feature_repo
feast apply
```

This registers feature definitions to Feast registry.

### 3. Materialize Features

Load features into online store (SQLite):

```bash
python feature_store/materialize_features.py --ticker BBCA.JK --days 30
```

### 4. Test Feature Retrieval

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

# Initialize store
store = FeatureStore(repo_path="feature_store/feature_repo")

# Fetch features for a ticker
entity_df = pd.DataFrame({
    "ticker": ["BBCA.JK"],
    "event_timestamp": [datetime.now()]
})

features = store.get_online_features(
    features=[
        "stock_technical_features:returns",
        "stock_technical_features:ma_5",
        "stock_technical_features:volatility_5"
    ],
    entity_rows=entity_df.to_dict('records')
).to_dict()

print(features)
```

## 🏗️ Architecture

### Feature Views

**`stock_technical_features`**:
- Entity: `ticker` (stock symbol)
- TTL: 1 day
- Features:
  - OHLCV: Open, High, Low, Close, Volume
  - Returns: Daily returns
  - Moving Averages: ma_5, ma_10, ma_20
  - Volatility: volatility_5, volatility_10, volatility_20

### Storage

- **Offline Store**: Parquet files (`data/processed/feast_features.parquet`)
- **Online Store**: SQLite (`data/online_store.db`) for development
- **Registry**: Local file (`data/registry.db`)

### Production Configuration

For production, update `feature_store.yaml`:

```yaml
online_store:
  type: redis
  connection_string: "redis:6379"
```

## 🔄 Workflow

### Training Pipeline

1. **Feature Engineering** → `data/processed/{ticker}_processed.csv`
2. **Convert to Parquet** → `data/processed/feast_features.parquet`
3. **Train Model** using offline features

### Serving Pipeline

1. **Materialize Features** → Load to online store
2. **API Request** → Fetch features from Feast
3. **Predict** → Return result

## 📊 Performance

- **Feature Retrieval**: < 10ms (SQLite), < 5ms (Redis)
- **Materialization**: ~1s per 1000 rows
- **Storage**: ~1MB per 10,000 rows (Parquet)

## 🛠️ Maintenance

### Update Features

After retraining with new data:

```bash
# 1. Prepare new data
python feature_store/materialize_features.py --ticker BBCA.JK --prepare-only

# 2. Materialize to online store
python feature_store/materialize_features.py --ticker BBCA.JK --days 30
```

### Monitor Feature Freshness

Check latest feature timestamp:

```python
from feast import FeatureStore
store = FeatureStore(repo_path="feature_store/feature_repo")

# Get feature metadata
print(store.list_feature_views())
```

## 🧪 Testing

Test Feast integration:

```bash
# Start API server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Run integration tests
python tests/test_api_integration.py
```

## 📚 Resources

- [Feast Documentation](https://docs.feast.dev/)
- [Feature Store Concepts](https://docs.feast.dev/getting-started/concepts)
- [Production Deployment](https://docs.feast.dev/how-to-guides/running-feast-in-production)

---

*Part of StockFlowML Level 3: Production Serving*
