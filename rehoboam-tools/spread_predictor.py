import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_price_data(csv_file, symA, symB):
    """Load and preprocess price data from CSV for specific symbols."""
    try:
        # Read CSV - let pandas infer the date format
        df = pd.read_csv(csv_file, parse_dates=['Date'])

        # Check required columns
        if 'Date' not in df.columns:
            raise ValueError(f"Required column 'Date' not found in CSV.")

        # Check if symbols exist
        if symA not in df.columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Symbol '{symA}' not found in CSV.")
        if symB not in df.columns:
            raise ValueError(f"Symbol '{symB}' not found in CSV.")

        # Validate Date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            print(f"Warning: {invalid_dates} invalid date entries found. Dropping them.")
            df = df.dropna(subset=['Date'])

        # Convert symbol columns to numeric, handling empty cells
        df[symA] = pd.to_numeric(df[symA], errors='coerce')
        df[symB] = pd.to_numeric(df[symB], errors='coerce')

        # Count missing values before dropping
        missing_a = df[symA].isna().sum()
        missing_b = df[symB].isna().sum()
        total_rows = len(df)

        print(f"Data quality check:")
        print(f"  - Total rows: {total_rows}")
        print(f"  - Missing {symA}: {missing_a} ({missing_a / total_rows * 100:.1f}%)")
        print(f"  - Missing {symB}: {missing_b} ({missing_b / total_rows * 100:.1f}%)")

        # Drop rows where BOTH symbols have data (for pairs trading)
        df_clean = df.dropna(subset=[symA, symB])
        df_clean = df_clean[(df_clean[symA] != 0) & (df_clean[symB] != 0)]

        print(f"  - Rows with valid data for both symbols: {len(df_clean)}")

        if df_clean.empty:
            raise ValueError("No valid data after filtering NaNs and zeros.")

        # Set Date as index
        df_clean = df_clean.set_index('Date')
        df_clean = df_clean.sort_index()  # Ensure chronological order

        # Check for duplicates
        duplicates = df_clean.index.duplicated().sum()
        if duplicates > 0:
            print(f"Warning: {duplicates} duplicate timestamps found. Keeping first occurrence.")
            df_clean = df_clean[~df_clean.index.duplicated(keep='first')]

        return df_clean[[symA, symB]]
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None


def calculate_spread_rolling_beta(df, symA, symB, window=720):  # 30 days
    """Calculate spread using rolling OLS for dynamic beta."""
    priceA = df[symA].values
    priceB = df[symB].values

    beta = np.zeros(len(priceA))

    for i in range(window, len(priceA)):
        X = add_constant(priceB[i - window:i])
        y = priceA[i - window:i]
        model = OLS(y, X).fit()
        beta[i] = model.params[1]

    # Forward fill initial beta values
    beta[:window] = beta[window] if len(priceA) > window else np.nan

    spread = priceA - beta * priceB
    return pd.Series(spread, index=df.index), beta


def calculate_ou_parameters(spread):
    """Calculate Ornstein-Uhlenbeck parameters correctly."""
    # Remove NaN values
    spread_clean = spread.dropna()

    if len(spread_clean) < 2:
        print("Warning: Insufficient data for O-U parameter estimation")
        return 0.01, spread_clean.mean() if len(spread_clean) > 0 else 0, 0.01

    # First difference
    spread_diff = np.diff(spread_clean.values)
    spread_lagged = spread_clean.values[:-1]

    # OLS: ΔS_t = α + β * S_{t-1} + ε
    # For O-U: ΔS_t = κ(θ - S_{t-1}) + σ * ε
    # So: β = -κ and α = κθ

    X = add_constant(spread_lagged)
    model = OLS(spread_diff, X).fit()

    beta_coef = model.params[1]
    alpha_coef = model.params[0]

    if beta_coef >= 0:
        # No mean reversion detected
        print("Warning: No mean reversion detected (beta >= 0)")
        kappa = 0.01
        theta = spread_clean.mean()
    else:
        kappa = -beta_coef
        theta = alpha_coef / kappa if kappa != 0 else spread_clean.mean()

    sigma = np.sqrt(model.mse_resid)

    return kappa, theta, sigma


def forecast_spread_ou(spread, forecast_steps=720):
    """Forecast spread using correct Ornstein-Uhlenbeck process."""
    kappa, theta, sigma = calculate_ou_parameters(spread)

    print(f"O-U Parameters - kappa: {kappa:.4f}, theta: {theta:.4f}, sigma: {sigma:.4f}")

    # Start from last observed value
    forecast = np.zeros(forecast_steps)
    current_value = spread.iloc[-1]

    dt = 1.0  # Time step (1 hour in your case)

    for i in range(forecast_steps):
        # Correct O-U process: dS = κ(θ - S)dt + σdW
        # Discrete: S_{t+1} = S_t + κ(θ - S_t)dt + σ√dt * Z
        drift = kappa * (theta - current_value) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
        current_value = current_value + drift + diffusion
        forecast[i] = current_value

    forecast_index = pd.date_range(
        start=spread.index[-1] + pd.Timedelta(hours=1),
        periods=forecast_steps,
        freq='H'
    )

    return pd.Series(forecast, index=forecast_index)


def calculate_half_life(kappa):
    """Calculate mean reversion half-life."""
    return np.log(2) / kappa if kappa > 0 else np.inf


def plot_spread(spread, forecast, symA, symB):
    """Plot historical spread and forecast."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Full historical spread with forecast
    ax1.plot(spread.index, spread.values, label='Historical Spread', color='blue', alpha=0.7)
    ax1.plot(forecast.index, forecast.values, label='Forecast', color='red', alpha=0.5)
    ax1.axhline(y=spread.mean(), color='green', linestyle='--', label='Historical Mean', alpha=0.5)
    ax1.set_title(f'Spread: {symA} - β×{symB}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Spread Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Last 30 days of spread + forecast
    last_30_days = spread.iloc[-720:] if len(spread) > 720 else spread
    ax2.plot(last_30_days.index, last_30_days.values, label='Last 30 Days', color='blue', linewidth=2)
    ax2.plot(forecast.index, forecast.values, label='30-Day Forecast', color='red', alpha=0.7)

    # Add confidence bands for forecast
    forecast_mean = forecast.mean()
    forecast_std = forecast.std()
    ax2.fill_between(forecast.index,
                     forecast_mean - 2 * forecast_std,
                     forecast_mean + 2 * forecast_std,
                     alpha=0.2, color='red', label='95% Confidence')

    ax2.axhline(y=spread.mean(), color='green', linestyle='--', label='Mean', alpha=0.5)
    ax2.set_title('Recent Spread & Forecast (30 Days)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Spread Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def suggest_parameters(historical_sigma, forecast_sigma, spread_mean=None):
    """Suggest trading parameters based on spread analysis."""
    # Combine historical and forecast volatility for more robust estimation
    combined_sigma = (historical_sigma + forecast_sigma) / 2

    # Entry Z-Score: typically between 1.0 and 2.0 standard deviations for FX pairs
    # FX pairs are generally more mean-reverting, so we can use tighter entry points
    volatility_ratio = forecast_sigma / historical_sigma if historical_sigma > 0 else 1

    if volatility_ratio > 1.2:  # Forecast is more volatile - be more conservative
        suggested_entry_z = 2.0
    elif volatility_ratio < 0.8:  # Forecast is less volatile - can be more aggressive
        suggested_entry_z = 1.0
    else:
        suggested_entry_z = 1.5

    # Stop Loss: For FX pairs, typically 2-3% is reasonable
    # Calculate as percentage of the mean spread value if available
    if spread_mean and spread_mean != 0:
        # Stop loss as 3-4 standard deviations relative to mean
        sl_in_spread_units = combined_sigma * 3.5
        suggested_sl_percent = abs((sl_in_spread_units / abs(spread_mean)) * 100)
        # Cap between 2% and 5% for FX pairs
        suggested_sl_percent = min(5.0, max(2.0, suggested_sl_percent))
    else:
        # Default conservative stop loss for FX
        suggested_sl_percent = 3.0

    return suggested_entry_z, suggested_sl_percent


# Usage with corrected logic
if __name__ == "__main__":
    csv_file = 'price_history.csv'
    symA, symB = 'GBPUSD', 'EURUSD'

    # Load data
    df = load_price_data(csv_file, symA, symB)
    if df is None:
        print("Failed to load data. Exiting.")
        exit(1)

    print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

    # Calculate spread with rolling beta
    spread, beta = calculate_spread_rolling_beta(df, symA, symB)

    # Remove NaN values from spread
    spread = spread.dropna()

    if len(spread) < 100:
        print("Warning: Insufficient data for reliable analysis")
        exit(1)

    # Calculate O-U parameters
    kappa, theta, sigma = calculate_ou_parameters(spread)

    # Calculate half-life
    half_life = calculate_half_life(kappa)
    print(f"\nMean reversion half-life: {half_life:.1f} hours ({half_life / 24:.1f} days)")

    # Generate forecast
    forecast = forecast_spread_ou(spread)

    # Calculate statistics
    historical_sigma = spread.std()
    forecast_sigma = forecast.std()

    # Plot results
    plot_spread(spread, forecast, symA, symB)

    # Suggest trading parameters
    suggested_entry_z, suggested_sl_percent = suggest_parameters(
        historical_sigma, forecast_sigma, spread.mean()
    )

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"TRADING PARAMETERS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Pair: {symA}/{symB}")
    print(f"Data points analyzed: {len(spread)}")
    print(f"Historical Spread Mean: {spread.mean():.6f}")
    print(f"Historical Spread Sigma: {historical_sigma:.6f}")
    print(f"Forecast Spread Sigma: {forecast_sigma:.6f}")
    print(f"Current Spread Value: {spread.iloc[-1]:.6f}")
    print(f"Mean Reversion Level (θ): {theta:.6f}")
    print(f"Mean Reversion Speed (κ): {kappa:.4f}")
    print(f"Suggested Entry Z-Score: ±{suggested_entry_z:.2f}σ")
    print(f"Suggested Stop Loss: {suggested_sl_percent:.1f}%")
    print(f"Current Z-Score: {(spread.iloc[-1] - theta) / historical_sigma:.2f}")

    # Trading signals
    entry_threshold = suggested_entry_z * historical_sigma
    print(f"\nTrading Thresholds:")
    print(f"Long Entry (spread < θ - {suggested_entry_z}σ): {theta - entry_threshold:.6f}")
    print(f"Short Entry (spread > θ + {suggested_entry_z}σ): {theta + entry_threshold:.6f}")
    print(f"Current Position: ", end="")
    current_z = (spread.iloc[-1] - theta) / historical_sigma
    if abs(current_z) < suggested_entry_z:
        print("NEUTRAL (within entry thresholds)")
    elif current_z > suggested_entry_z:
        print(f"SHORT SIGNAL (spread is {current_z:.2f}σ above mean)")
    else:
        print(f"LONG SIGNAL (spread is {abs(current_z):.2f}σ below mean)")
    print(f"{'=' * 50}")