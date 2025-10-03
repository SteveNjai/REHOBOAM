import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configurable variables
GBP_CSV_FILE = "GBPUSD_M1_price_history.csv"
EUR_CSV_FILE = "EURUSD_M1_price_history.csv"
BASE_DIR = os.getcwd()
HISTORY_DIR = os.path.join(BASE_DIR, "history")
ROLLING_WINDOW_VOL = 20  # Window size for volatility calculation (minutes)
BASE_WINDOW_HALF = 20  # Window size for half-life calculation (minutes)
REGRESSION_WINDOW = 252  # Window size for hedge ratio regression (minutes)
MIN_PERIODS_HALF = 2  # Minimum periods for half-life rolling window
VOL_THRESHOLD = 0.01  # Volatility threshold for high volatility adjustment (adjusted from 0.0005)
VOL_SCALE = 0.5  # Scale factor for volatility adjustment (per 0.001 above threshold)
entry_zscore_multiplier = 0.8   #scale factor for entry zscore adjust according to backtested data
stoploss_zscore_multiplier = 1.6    #scale factor for stoploss zscore. adjust according to backtested data

#output directory
OUTPUT_DIR = "results"  # Relative path to the folder
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_price_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    df['DATETIME'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    df.set_index('DATETIME', inplace=True)
    print(f"Loaded {len(df)} bars from {file_path}")
    return df

def calculate_spread(df_gbp, df_eur):
    common_index = df_gbp.index.intersection(df_eur.index)
    df_gbp = df_gbp.loc[common_index].copy()
    df_eur = df_eur.loc[common_index].copy()

    # Calculate returns
    df_gbp['RETURNS'] = df_gbp['<CLOSE>'].pct_change()
    df_eur['RETURNS'] = df_eur['<CLOSE>'].pct_change()

    # Initialize hedge ratio series
    hedge_ratios = []
    for i in range(len(df_gbp) - REGRESSION_WINDOW + 1):
        window_gbp = df_gbp['RETURNS'].iloc[i:i + REGRESSION_WINDOW].dropna()
        window_eur = df_eur['RETURNS'].iloc[i:i + REGRESSION_WINDOW].dropna()
        if len(window_gbp) >= 2 and len(window_eur) >= 2:
            slope, _ = np.polyfit(window_gbp, window_eur, 1)
            hedge_ratios.append(slope)
        else:
            hedge_ratios.append(np.nan)
    hedge_ratios = pd.Series(hedge_ratios, index=df_gbp.index[REGRESSION_WINDOW-1:])

    # Forward-fill hedge ratios and set initial values
    hedge_ratios = hedge_ratios.ffill().reindex(df_gbp.index, method='ffill')
    hedge_ratios.iloc[:REGRESSION_WINDOW-1] = hedge_ratios.iloc[REGRESSION_WINDOW-1]

    # Calculate spread using dynamic hedge ratio
    df_gbp['SPREAD_ADJ'] = df_gbp['<CLOSE>'] - (hedge_ratios * df_eur['<CLOSE>'])
    print(f"Spread variation: {df_gbp['SPREAD_ADJ'].std()}")
    print(f"Average Hedge Ratio: {hedge_ratios.mean():.4f}")
    return df_gbp, hedge_ratios

def calculate_volatility(spread_series, window=ROLLING_WINDOW_VOL):
    return spread_series.rolling(window=window, min_periods=1).std()

def calculate_half_life(spread_series):
    # Simplified approximation using autocorrelation lag-1
    spread_series = spread_series.dropna()
    if len(spread_series) < MIN_PERIODS_HALF:
        return np.inf
    try:
        autocorr = spread_series.autocorr(lag=1)
        if -1 < autocorr < 1:
            half_life = -np.log(1.0 - autocorr) / np.log(2.0) if autocorr < 1 else np.inf
        else:
            half_life = np.inf
    except:
        half_life = np.inf
    return half_life

def predict_zscore(volatility, half_life):
    base_zscore = 3.0
    if pd.notna(volatility) and volatility > VOL_THRESHOLD:
        zscore = max(4.0, base_zscore + VOL_SCALE * (volatility - VOL_THRESHOLD))
    elif pd.notna(half_life) and half_life < 1.0:
        zscore = max(1.5, base_zscore - (1.0 - half_life))
    else:
        zscore = base_zscore
    return min(7.0, max(1.0, zscore))

def analyze_and_plot(file_gbp_path, file_eur_path):
    start_time = time.time()
    print(f"Starting analysis at {time.ctime(start_time)}")

    df_gbp = load_price_data(file_gbp_path)
    df_eur = load_price_data(file_eur_path)

    df_gbp, hedge_ratios = calculate_spread(df_gbp, df_eur)

    volatility = calculate_volatility(df_gbp['SPREAD_ADJ'], window=ROLLING_WINDOW_VOL)
    half_life = df_gbp['SPREAD_ADJ'].rolling(window=BASE_WINDOW_HALF, min_periods=MIN_PERIODS_HALF).apply(calculate_half_life, raw=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(volatility, df_gbp['SPREAD_ADJ'], c=volatility, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Volatility')
    plt.xlabel('Volatility (Rolling Std Dev)')
    plt.ylabel('Spread')
    plt.title('Spread vs Volatility')

    plt.subplot(1, 2, 2)
    plt.scatter(half_life, df_gbp['SPREAD_ADJ'], c=half_life, cmap='plasma', alpha=0.5)
    plt.colorbar(label='Half-Life (Hours)')
    plt.xlabel('Half-Life')
    plt.ylabel('Spread')
    plt.title('Spread vs Half-Life')

    plt.tight_layout()

    #SAVE THE figure
    full_path = os.path.join(OUTPUT_DIR, 'spread.png')
    plt.savefig(full_path)

    z_scores = [predict_zscore(v, h) for v, h in zip(volatility, half_life)]
    df_gbp['PREDICTED_ZSCORE'] = z_scores

    # Calculate recommended entry and stop-loss z-scores
    df_gbp['ENTRY_ZSCORE'] = (df_gbp['PREDICTED_ZSCORE']*entry_zscore_multiplier).clip(lower = 1.5)
    df_gbp['STOPLOSS_ZSCORE'] = (df_gbp['PREDICTED_ZSCORE'] * stoploss_zscore_multiplier).clip(upper=10.0)

    print("Predicted Z-Scores over time and trade reccomendations:")
    print(df_gbp[['<CLOSE>', 'SPREAD_ADJ', 'PREDICTED_ZSCORE',
                  'ENTRY_ZSCORE', 'STOPLOSS_ZSCORE']].tail())
    plt.figure(figsize=(12, 4))
    plt.plot(df_gbp.index, df_gbp['PREDICTED_ZSCORE'], label='Predicted Z-Score')
    plt.xlabel('Time')
    plt.ylabel('Predicted Z-Score')
    plt.title('Predicted Z-Score Over Time')
    plt.legend()

    # save the figure
    full_path = os.path.join(OUTPUT_DIR, 'pedicted_zscore')
    plt.savefig(full_path)

    #plot all figures
    plt.show()

    #save the predicted figures

    output_file = os.path.join(OUTPUT_DIR, 'predicted_zscores.csv')
    df_gbp[['<CLOSE>', 'SPREAD_ADJ', 'PREDICTED_ZSCORE',
                  'ENTRY_ZSCORE', 'STOPLOSS_ZSCORE']].to_csv(output_file, sep=',')
    print(f"Results saved to {output_file}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    gbp_file = os.path.join(HISTORY_DIR, GBP_CSV_FILE)
    eur_file = os.path.join(HISTORY_DIR, EUR_CSV_FILE)
    analyze_and_plot(gbp_file, eur_file)