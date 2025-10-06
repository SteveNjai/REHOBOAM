import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Load CSV (columns: Time, EURUSD_Close, GBPUSD_Close)
df = pd.read_csv('prices_export.csv')
df['Spread'] = df['EURUSD_Close'] - beta * df['GBPUSD_Close']  # Beta from prior calc

# Rolling Z (Lookback=20)
df['Mu_S'] = df['Spread'].rolling(20).mean()
df['Sigma_S'] = df['Spread'].rolling(20).std()
df['Z'] = (df['Spread'] - df['Mu_S']) / df['Sigma_S']

# Stats
z_vals = df['Z'].dropna()
print(f"Mean: {z_vals.mean():.6f}, Std: {z_vals.std():.3f}")
print(f"Skew: {stats.skew(z_vals):.3f}, Kurt: {stats.kurtosis(z_vals):.3f}")
print(f"95th Percentile (|Z|): {z_vals.abs().quantile(0.95):.2f}")  # Suggested EntryZScore

# Adjust: EntryZ = 95th + 0.2 buffer
entry_z = z_vals.abs().quantile(0.95) + 0.2
print(f"Recommended EntryZScore: {entry_z:.2f}")