import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from sklearn.linear_model import LinearRegression


def fetch_and_compute_zscores(SymbolA, SymbolB, start_date_str, end_date_str, lookback=20, regression_period=252):
    """
    Fetches M1 bar data directly from MT5 for two symbols, computes Z-scores of the spread using arithmetic mean and standard deviation,
    plots histogram and time series, and computes statistics.

    Parameters:
    - SymbolA, SymbolB: str, e.g., 'XAUUSD', 'XAGUSD'
    - start_date_str, end_date_str: str, format 'YYYY-MM-DD'
    - lookback: int, rolling window for mu/sigma
    - regression_period: int, number of recent M1 bars for Beta calculation

    Requires MT5 running and Python integration setup.
    """
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    start = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Fetch M1 bars directly (more efficient than ticks + resample)
    ratesA = mt5.copy_rates_range(SymbolA, mt5.TIMEFRAME_M1, start, end)
    ratesB = mt5.copy_rates_range(SymbolB, mt5.TIMEFRAME_M1, start, end)

    if len(ratesA) == 0 or len(ratesB) == 0:
        print("No M1 bar data fetched")
        mt5.shutdown()
        return

    # Convert to DataFrame using close prices
    dfA = pd.DataFrame(ratesA)
    dfA['time'] = pd.to_datetime(dfA['time'], unit='s')
    dfA = dfA.set_index('time')['close'].rename('A')

    dfB = pd.DataFrame(ratesB)
    dfB['time'] = pd.to_datetime(dfB['time'], unit='s')
    dfB = dfB.set_index('time')['close'].rename('B')

    # Merge on time index
    df = pd.concat([dfA, dfB], axis=1).dropna()

    # Compute Beta on recent M1 data
    if len(df) < regression_period:
        print("Insufficient data for regression")
        mt5.shutdown()
        return
    m1_data = df.tail(regression_period)
    reg = LinearRegression().fit(m1_data['B'].values.reshape(-1, 1), m1_data['A'].values)
    beta = reg.coef_[0]
    print(f"Calculated Beta on M1 data: {beta}")

    # Compute spread
    df['spread'] = df['A'] - beta * df['B']

    # Compute rolling mu (arithmetic mean) and sigma (standard deviation)
    df['mu'] = df['spread'].rolling(lookback).mean()
    df['sigma'] = df['spread'].rolling(lookback).std()

    # Z-score
    df['zscore'] = (df['spread'] - df['mu']) / df['sigma']
    df.dropna(inplace=True)

    if df.empty:
        print("No valid Z-scores computed")
        mt5.shutdown()
        return

    Z = df['zscore']

    # Statistics
    mean = Z.mean()
    median = Z.median()
    mode = Z.mode()[0] if not Z.mode().empty else np.nan
    std = Z.std()
    skewness = stats.skew(Z)
    kurtosis = stats.kurtosis(Z)
    range_val = Z.max() - Z.min()
    sem = std / np.sqrt(len(Z))

    print("\nZ-score Statistics:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")
    print(f"Range: {range_val}")
    print(f"Standard Error: {sem}")

    # Plots
    plt.figure(figsize=(10, 5))
    Z.hist(bins=50)
    plt.title('Histogram of Z-scores')
    plt.xlabel('Z-score')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 5))
    Z.plot()
    plt.title('Time Series of Z-scores')
    plt.xlabel('Time')
    plt.ylabel('Z-score')
    plt.show()

    mt5.shutdown()

#-------------------
# USAGE
#---------------
#ask for user input
print ("WELCOME TO Z-SCORE ANALYZER")
print("\nEnter SymbolA (e.g. GBPUSD): ")
SymbolA = input()
print("\nEnter SymbolB (e.g. EURUSD): ")
SymbolB = input()
print("\nEnter Start Date (yyy-mm-dd): ")
start_date = input()
print("\nEnter End date (yyyy-mm-dd): ")
end_date = input()
print(f"\n RUNNING Z-SCORE ANALYSIS FOR SYMBOL {SymbolA} and {SymbolB} strating from {start_date} to {end_date}...\n")

#run function
fetch_and_compute_zscores(SymbolA, SymbolB, start_date, end_date, lookback=20, regression_period=252)