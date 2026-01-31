"""
Generate realistic fallback snapshot data for BBCA.JK
"""
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 200 trading days (roughly 10 months)
dates = pd.date_range('2021-01-04', periods=250, freq='D')
# Filter weekdays only
dates = [d for d in dates if d.weekday() < 5][:200]

# Generate realistic price series with random walk
opens = [32000]
for i in range(len(dates)-1):
    # Random walk with drift
    change = np.random.normal(20, 300)  # slight upward drift, high volatility
    new_open = opens[-1] + change
    # Keep prices reasonable
    opens.append(max(min(new_open, 40000), 28000))

df = pd.DataFrame({'Date': dates})
df['Open'] = opens

# Generate Close prices (can be higher or lower than Open)
df['Close'] = df['Open'] + np.random.normal(0, 250, len(df))

# High is max of Open/Close + some positive noise
df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(50, 300, len(df))

# Low is min of Open/Close minus some positive noise
df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(50, 300, len(df))

# Ensure price integrity (High >= Low)
df['High'] = df[['High', 'Low']].max(axis=1) + 100
df['Low'] = df[['High', 'Low']].min(axis=1) - 100

# Volume: realistic trading volume
df['Volume'] = np.random.randint(15000000, 30000000, len(df))

# Convert Date to string for CSV
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

# Save
output_path = 'data/fallback/BBCA_JK_snapshot.csv'
df.to_csv(output_path, index=False)

print(f"✓ Created {len(df)} rows of realistic stock data")
print(f"✓ Saved to {output_path}")
print(f"\nPrice range: {df['Close'].min():.0f} - {df['Close'].max():.0f}")
print(f"Average volume: {df['Volume'].mean()/1e6:.1f}M")

# Check up/down distribution
df['Close_num'] = pd.to_numeric(df['Close'])
ups = (df['Close_num'].diff() > 0).sum()
downs = (df['Close_num'].diff() < 0).sum()
print(f"\nDaily movements: {ups} ups, {downs} downs ({ups/(ups+downs)*100:.1f}% up days)")

print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())
