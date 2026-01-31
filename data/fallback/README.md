# BBCA_JK Snapshot Metadata

**Ticker**: BBCA.JK (Bank Central Asia)  
**Generated**: 2026-01-30  
**Purpose**: Fallback data for operational continuity when Yahoo Finance API is unavailable  
**Status**: SAMPLE/SNAPSHOT data for validation - NOT real-time market data

## Data Characteristics

- **Sample Period**: Feb-Mar 2021 (30 trading days)
- **Price Range**: 32,400 - 35,700 IDR
- **Average Volume**: ~22.5M shares/day
- **Source**: Synthetic realistic data for testing

## Usage

This snapshot is automatically used by the pipeline when Yahoo Finance fails. To update with real data:

```bash
# Download fresh data and save as snapshot
python -c "import yfinance as yf; df = yf.Ticker('BBCA.JK').history(period='5y'); df.reset_index().to_csv('data/fallback/BBCA_JK_snapshot.csv', index=False)"
```

**Important**: Always verify snapshot data before production use.
