import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller, coint
import json
import time
from datetime import datetime, timedelta
import os
import sys

# Change to allow user input
print("\nWELCOME TO ORACLE SPREAD OPTIMIZER TOOL.")

# Get symbols
if len(sys.argv) > 1:
    SYMBOL_A = sys.argv[1]
    print("SYMBOL A = ", SYMBOL_A)
    SYMBOL_B = sys.argv[2]
    print("SYMBOL B = ", SYMBOL_B)
else:
    print("No command-line argument provided. CHOOSING DEFAULT VALUES...")
    SYMBOL_A = 'GBPUSD'
    SYMBOL_B = 'EURUSD'

print("The rest of the inputs are hard coded in the python file. Edit that if you need them changed...")
print("\tNOTE: The parameters MUST match what is in the MT5 expert advisor inputs..\n")

# CREATE RESULTS AND OPTIMAL_ZSCORE DIRECTORIES IN THE CURRENT DIRECTORY
optimal_zscore_dir = os.path.join(os.getcwd(), "optimal_zscores")
results_dir = os.path.join(os.getcwd(), 'results')

try:
    os.makedirs(optimal_zscore_dir, exist_ok=True)
    print(f"Directory '{optimal_zscore_dir}' ensured to exist.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    os.makedirs(results_dir, exist_ok=True)
    print(f"Directory '{results_dir}' ensured to exist.")
except Exception as e:
    print(f"An error occurred: {e}")

# Configuration
RESAMPLING = 2  # Resampling period in seconds (e.g., 2 for 2s bars, 60 for 1m bars)
LOOKBACK_PERIOD = 5720  # Number of resampled bars for mean and std dev of spread
REGRESSION_PERIOD = 7020  # Number of resampled bars for hedge ratio calculation
STOP_ZSCORE = 4.8
RISK_REWARD_RATIO = 2.0
N_PATHS = 200
N_STEPS = 3600  # ~2hrs at RESAMPLING seconds per bar
MIN_CORRELATION = 0.2
MIN_COINT_PVALUE = 0.05
BYPASS_COINT_CHECK = True
BYPASS_CORR_CHECK = False
SHARED_FILE = f"{SYMBOL_A}-{SYMBOL_B}-optimal_zscore.txt"
ZSCORE_MAX = 5.0
LOT_SIZE = 0.1
PIP_VALUE = 10.0  # USD per pip for 1 lot
PIP_SIZE = 0.0001

# ------------------------------
# MQL5 COMMONS DIRECTORY
# ----------------------
# Build path to MetaTrader Common Files directory
common_files = os.path.expandvars(r"%AppData%\MetaQuotes\Terminal\Common\Files")

# Ensure directory exists
os.makedirs(common_files, exist_ok=True)

# ------------------
# INITIALIZE
# --------------------
def init_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    print("MT5 initialized")
    return True

# Custom covariance function
def custom_covariance(x, y):
    if len(x) != len(y) or len(x) == 0:
        print(f"Error: Arrays have different sizes ({len(x)} vs {len(y)}) or are empty")
        return 0.0
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)

# Calculate hedge ratio and Z-scores
def get_spread_data(symbol_a, symbol_b):
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(seconds=REGRESSION_PERIOD * RESAMPLING)
    retries = 3
    for attempt in range(retries):
        ticks_a = mt5.copy_ticks_range(symbol_a, utc_from, utc_to, mt5.COPY_TICKS_ALL)
        ticks_b = mt5.copy_ticks_range(symbol_b, utc_from, utc_to, mt5.COPY_TICKS_ALL)
        if ticks_a is not None and ticks_b is not None and len(ticks_a) >= REGRESSION_PERIOD * 0.8 and len(
                ticks_b) >= REGRESSION_PERIOD * 0.8:
            break
        print(
            f"Attempt {attempt + 1}/{retries}: Insufficient tick data for {symbol_a} ({len(ticks_a)}) or {symbol_b} ({len(ticks_b)}). Retrying.")
        utc_from -= timedelta(seconds=REGRESSION_PERIOD * RESAMPLING)
        time.sleep(1)
    else:
        print(f"Insufficient tick data after {retries} attempts. Falling back to M1 data.")
        rates_a = mt5.copy_rates_range(symbol_a, mt5.TIMEFRAME_M1, utc_from, utc_to)
        rates_b = mt5.copy_rates_range(symbol_b, mt5.TIMEFRAME_M1, utc_from, utc_to)
        if rates_a is None or rates_b is None or len(rates_a) < REGRESSION_PERIOD // 60 or len(
                rates_b) < REGRESSION_PERIOD // 60:
            print(f"Insufficient M1 data: {len(rates_a)} bars for {symbol_a}, {len(rates_b)} bars for {symbol_b}")
            return None, None, None, None
        df_a = pd.DataFrame(rates_a)[['time', 'close']].set_index('time')
        df_b = pd.DataFrame(rates_b)[['time', 'close']].set_index('time')
        df_a.index = pd.to_datetime(df_a.index, unit='s')
        df_b.index = pd.to_datetime(df_b.index, unit='s')
        print(f"M1 data: {len(df_a)} bars for {symbol_a}, {len(df_b)} bars for {symbol_b}")
        return process_data(df_a, df_b)

    # Process tick data and resample to RESAMPLING seconds
    df_ticks_a = pd.DataFrame(ticks_a)[['time_msc', 'bid', 'ask']]
    df_ticks_b = pd.DataFrame(ticks_b)[['time_msc', 'bid', 'ask']]
    df_ticks_a['time'] = pd.to_datetime(df_ticks_a['time_msc'], unit='ms')
    df_ticks_b['time'] = pd.to_datetime(df_ticks_b['time_msc'], unit='ms')
    df_ticks_a.set_index('time', inplace=True)
    df_ticks_b.set_index('time', inplace=True)
    df_resampled_a = df_ticks_a.resample(f'{RESAMPLING}s').last().ffill().dropna()
    df_resampled_b = df_ticks_b.resample(f'{RESAMPLING}s').last().ffill().dropna()
    df_a = pd.DataFrame({'close': (df_resampled_a['bid'] + df_resampled_a['ask']) / 2})
    df_b = pd.DataFrame({'close': (df_resampled_b['bid'] + df_resampled_b['ask']) / 2})
    print(f"Tick data: {len(df_a)} {RESAMPLING}s bars for {symbol_a} from {df_a.index[0]} to {df_a.index[-1]}")
    print(f"Tick data: {len(df_b)} {RESAMPLING}s bars for {symbol_b} from {df_b.index[0]} to {df_b.index[-1]}")
    return process_data(df_a, df_b)

# Process data (tick or M1)
def process_data(df_a, df_b):
    common_idx = df_a.index.intersection(df_b.index)
    if len(common_idx) < REGRESSION_PERIOD * 0.8:
        print(f"Insufficient overlapping data: {len(common_idx)} periods (need {REGRESSION_PERIOD * 0.8})")
        return None, None, None, None
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]
    print(f"Aligned data: {len(df_a)} {RESAMPLING}s bars for {SYMBOL_A}, {len(df_b)} {RESAMPLING}s bars for {SYMBOL_B}")

    # Cointegration test
    coint_pvalue = coint(df_a['close'], df_b['close'])[1]
    print(f"Cointegration p-value: {coint_pvalue:.4f}")
    if coint_pvalue > MIN_COINT_PVALUE and not BYPASS_COINT_CHECK:
        print(f"Pair not cointegrated (p-value {coint_pvalue:.4f} > {MIN_COINT_PVALUE}). Try AUDUSD/NZDUSD.")
        return None, None, None, None
    elif coint_pvalue > MIN_COINT_PVALUE:
        print(
            f"Warning: Pair not cointegrated (p-value {coint_pvalue:.4f} > {MIN_COINT_PVALUE}), but proceeding due to BYPASS_COINT_CHECK=True")
        if coint_pvalue > 0.5:
            print("Warning: High cointegration p-value. Consider AUDUSD/NZDUSD for better cointegration.")

    # Calculate Beta
    returns_a = df_a['close'].pct_change().dropna()
    returns_b = df_b['close'].pct_change().dropna()
    common_idx = returns_a.index.intersection(returns_b.index)
    if len(common_idx) < REGRESSION_PERIOD * 0.8 - 1:
        print(f"Insufficient overlapping returns: {len(common_idx)} periods")
        return None, None, None, None
    returns_a = returns_a.loc[common_idx]
    returns_b = returns_b.loc[common_idx]
    cov = custom_covariance(returns_a.values, returns_b.values)
    var_b = np.var(returns_b)
    correlation = cov / (returns_a.std() * returns_b.std()) if returns_a.std() * returns_b.std() != 0 else 0
    print(f"Pair correlation: {correlation:.4f}")
    if correlation < MIN_CORRELATION and not BYPASS_CORR_CHECK:
        print(f"Correlation {correlation:.4f} below threshold {MIN_CORRELATION}")
        return None, None, None, None
    if correlation < 0.5:
        print(f"Warning: Correlation {correlation:.4f} is low. Consider AUDUSD/NZDUSD for better cointegration.")
    beta = cov / var_b if var_b != 0 else 0
    print(f"Beta (hedge ratio): {beta:.4f}")

    # Calculate spread and Z-scores
    recent_a = df_a['close'][-LOOKBACK_PERIOD - 1:]
    recent_b = df_b['close'][-LOOKBACK_PERIOD - 1:]
    spreads = recent_a - beta * recent_b
    mu = spreads[:-1].mean()
    sigma = spreads[:-1].std()
    if sigma == 0 or np.isnan(sigma):
        print("Zero or NaN spread standard deviation")
        return None, None, None, None
    zscores = (spreads - mu) / sigma
    # Spread stats
    spread_skew = skew(spreads[:-1], nan_policy='omit') if not np.all(spreads[:-1] == spreads[:-1].iloc[0]) else 0.0
    spread_kurt = kurtosis(spreads[:-1], nan_policy='omit') if not np.all(spreads[:-1] == spreads[:-1].iloc[0]) else 0.0
    print(
        f"Historical Spread Stats: Mean={mu:.6f}, Std={sigma:.6f}, Skew={spread_skew:.4f}, Kurtosis={spread_kurt:.4f}")

    return zscores, mu, sigma, beta

# Histogram-based Z-Score simulation
def simulate_histogram(zscores, n_steps, n_paths):
    zscores = zscores[np.abs(zscores) < ZSCORE_MAX]  # Filter outliers
    if len(zscores) == 0:
        print("No valid Z-scores for histogram simulation")
        return np.zeros((n_paths, n_steps))
    paths = np.zeros((n_paths, n_steps))
    for t in range(n_steps):
        paths[:, t] = np.clip(np.random.choice(zscores, size=n_paths), -ZSCORE_MAX, ZSCORE_MAX)
    print(f"Simulated Z-Score Range: Min={paths.min():.2f}, Max={paths.max():.2f}")
    return paths

# Simulate portfolio effect with multiple trades per path
def simulate_portfolio(paths, mu, sigma, entry_zscores, stop_zscore, risk_reward_ratio):
    results = {}
    equity_curves = {z: [] for z in entry_zscores}  # Store equity curves for each Z-Score
    for entry_z in entry_zscores:
        pnls = []
        total_trades = 0
        for path_idx in range(len(paths)):
            path_pnl = 0.0
            path_trades = 0
            position = 0
            equity = [0.0] * N_STEPS  # Initialize equity curve at $0
            for t in range(1, len(paths[path_idx])):
                z = paths[path_idx, t]
                if position == 0:
                    if z <= -entry_z:  # Long spread
                        position = 1
                        initial_spread = mu + z * sigma
                        entry_z_actual = z
                        sl_z = entry_z_actual - (stop_zscore - entry_z)
                        tp_z = entry_z_actual + risk_reward_ratio * (stop_zscore - entry_z)
                        path_trades += 1
                    elif z >= entry_z:  # Short spread
                        position = -1
                        initial_spread = mu + z * sigma
                        entry_z_actual = z
                        sl_z = entry_z_actual + (stop_zscore - entry_z)
                        tp_z = entry_z_actual - risk_reward_ratio * (stop_zscore - entry_z)
                        path_trades += 1
                elif position != 0:
                    spread = mu + paths[path_idx, t] * sigma
                    if position == 1 and (paths[path_idx, t] <= sl_z or paths[path_idx, t] >= tp_z):
                        profit = (spread - initial_spread) / PIP_SIZE * PIP_VALUE * LOT_SIZE
                        path_pnl += profit
                        equity[t:] = [e + profit for e in equity[t:]]  # Update equity curve
                        position = 0
                        total_trades += 1
                    elif position == -1 and (paths[path_idx, t] >= sl_z or paths[path_idx, t] <= tp_z):
                        profit = (initial_spread - spread) / PIP_SIZE * PIP_VALUE * LOT_SIZE
                        path_pnl += profit
                        equity[t:] = [e + profit for e in equity[t:]]  # Update equity curve
                        position = 0
                        total_trades += 1
            if position != 0:
                spread = mu + paths[path_idx, -1] * sigma
                profit = (spread - initial_spread) * position / PIP_SIZE * PIP_VALUE * LOT_SIZE
                path_pnl += profit
                equity[-1] = equity[-1] + profit
                total_trades += 1
            pnls.append(path_pnl)
            total_trades += path_trades - (1 if position != 0 else 0)
            if path_trades > 0:
                equity_curves[entry_z].append(equity)  # Store equity curve if trades occurred
        mean_pnl = np.mean(pnls) if pnls else 0
        std_pnl = np.std(pnls) if pnls else 0
        sharpe = mean_pnl / std_pnl if std_pnl != 0 else 0
        avg_trades_per_path = total_trades / N_PATHS if N_PATHS > 0 else 0
        results[entry_z] = {'sharpe': sharpe, 'mean_pnl': mean_pnl, 'std_pnl': std_pnl, 'trades': total_trades,
                            'avg_trades_per_path': avg_trades_per_path}
        print(
            f"Entry Z-Score {entry_z:.1f}: {total_trades} trades, Avg Trades/Path: {avg_trades_per_path:.2f}, Sharpe: {sharpe:.4f}, Mean PNL: {mean_pnl:.2f} USD")
    return results, equity_curves

# Plot results
def plot_results(paths, results, equity_curves, mu, entry_zscores, zscores):
    # Portfolio effect
    plt.figure(figsize=(12, 8))
    plt.plot(entry_zscores, [results[z]['sharpe'] for z in entry_zscores], label='Sharpe Ratio', color='#4CAF50')
    plt.xlabel("Entry Z-Score")
    plt.ylabel("Sharpe Ratio")
    plt.title(f"Portfolio Effect by Entry Z-Score ({RESAMPLING}s)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, f"{SYMBOL_A}-{SYMBOL_B}-portfolio_effect.png"))
    plt.close()

    # Histogram of historical vs. simulated Z-Scores
    plt.figure(figsize=(10, 5))
    plt.hist(zscores, bins=40, alpha=0.5, label='Historical Z', density=True)
    plt.hist(paths.flatten(), bins=40, alpha=0.5, label='Simulated Z (Histogram)', density=True)
    plt.legend()
    plt.title("Historical vs. Simulated Histogram Z-Score Distribution")
    plt.xlabel("Z-Score")
    plt.ylabel("Density")
    plt.grid()
    plt.savefig(os.path.join(results_dir, f"{SYMBOL_A}-{SYMBOL_B}-zscore_histogram.png"))
    plt.close()

    # Equity curves for each Z-Score with trades
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(entry_zscores)))
    for idx, z in enumerate(entry_zscores):
        if results[z]['trades'] > 0 and equity_curves[z]:
            # Average equity curve across paths with trades
            avg_equity = np.mean(equity_curves[z], axis=0)
            plt.plot(range(N_STEPS), avg_equity, label=f'Z={z:.1f}', color=colors[idx], alpha=0.7)
    plt.xlabel(f"Steps ({RESAMPLING}s intervals)")
    plt.ylabel("Equity (USD)")
    plt.title("Average Equity Curves by Entry Z-Score (Initial Equity = $0)")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{SYMBOL_A}-{SYMBOL_B}-equity_curves.png"))
    plt.close()

# Write optimal Z-score to shared file
def write_optimal_z(results):
    try:
        max_sharpe = -float('inf')
        optimal_z = None
        for z in results:
            if results[z]['mean_pnl'] > 0 and results[z]['sharpe'] > max_sharpe:
                max_sharpe = results[z]['sharpe']
                optimal_z = z
        if optimal_z is None:
            max_pnl = -float('inf')
            for z in results:
                if results[z]['mean_pnl'] > max_pnl:
                    max_pnl = results[z]['mean_pnl']
                    optimal_z = round(z, 1)
        if optimal_z is None:
            optimal_z = min(results.keys())

        # Save to both directories
        zscore_filename = f"{SYMBOL_A}-{SYMBOL_B}-optimal_zscore.txt"
        for file_path in [
            os.path.join(optimal_zscore_dir, zscore_filename),
            os.path.join(common_files, zscore_filename)
        ]:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"{optimal_z:.1f}")
            print(f"\nWrote optimal Z-score {optimal_z:.1f} to {file_path}")
        return optimal_z

    except Exception as e:
        print(f"Error writing optimal Z-score: {str(e)}")
        return 0.0

# Main function
def main():
    if not init_mt5():
        return
    zscores, mu, sigma, beta = get_spread_data(SYMBOL_A, SYMBOL_B)
    if zscores is None:
        mt5.shutdown()
        return

    # Z-Score stats
    z_skew = skew(zscores, nan_policy='omit') if not np.all(zscores == zscores.iloc[0]) else 0.0
    z_kurt = kurtosis(zscores, nan_policy='omit') if not np.all(zscores == zscores.iloc[0]) else 0.0
    print(
        f"\nHistorical Z-Score Stats: Mean={zscores.mean():.4f}, Std={zscores.std():.4f}, Skew={z_skew:.4f}, Kurtosis={z_kurt:.4f}")

    # Simulate paths using historical Z-Score histogram
    paths = simulate_histogram(zscores, N_STEPS, N_PATHS)

    # Simulate portfolio
    entry_zscores = np.arange(0, ZSCORE_MAX, 0.2)
    results, equity_curves = simulate_portfolio(paths, mu, sigma, entry_zscores, STOP_ZSCORE, RISK_REWARD_RATIO)

    # Plot
    plot_results(paths, results, equity_curves, zscores.mean(), entry_zscores, zscores)

    # Find and write optimal Z-score
    optimal_z = write_optimal_z(results)
    print(
        f"\nOptimal Entry Z-Score: {optimal_z:.1f}, Sharpe: {results[optimal_z]['sharpe']:.4f}, Mean PNL: {results[optimal_z]['mean_pnl']:.2f} USD, Avg Trades/Path: {results[optimal_z]['avg_trades_per_path']:.2f}")

    mt5.shutdown()

if __name__ == "__main__":
    while True:
        try:
            main()
            time.sleep(60)
        except KeyboardInterrupt:
            print("Script interrupted by user")
            mt5.shutdown()
            break