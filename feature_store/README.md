# Level 3 Placeholder: Feature Store

This directory will contain Feast feature store configuration for Level 3.

## Planned Structure

```
feature_store/
├── feature_repo/
│   ├── features.py          # Feature definitions
│   └── feature_store.yaml   # Feast configuration
└── materialize_features.py  # Materialization script
```

## Components (Planned)

- **Feature Views**: Define stock features for online/offline serving
- **Entities**: Stock ticker as entity
- **Online Store**: Redis for low-latency feature retrieval
- **Offline Store**: Parquet files for training

## Tech Stack (Level 3)

- Feast for feature store
- Redis for online storage
- Parquet for offline storage

---

*To be implemented in Level 3*
