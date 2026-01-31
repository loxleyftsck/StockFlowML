import pandas as pd

# Load and check
df = pd.read_csv('data/fallback/BBCA_JK_snapshot.csv', parse_dates=['Date'])
print(f"Loaded {len(df)} rows")

# Calculate target
df['Close_next'] = df['Close'].shift(-1)
df['target'] = (df['Close_next'] > df['Close']).astype(int)

print(f"\nTarget distribution:")
print(df['target'].value_counts())
print(f"\nTarget mean: {df['target'].mean():.2%}")

print(f"\nFirst 10 rows:")
print(df[['Date', 'Close', 'Close_next', 'target']].head(10))

print(f"\nLast 10 rows:")
print(df[['Date', 'Close', 'Close_next', 'target']].tail(10))
