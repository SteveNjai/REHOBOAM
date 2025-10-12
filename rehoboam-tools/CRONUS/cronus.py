import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller, coint
import time
from datetime import datetime, timedelta
import os
import sys
import signal
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === User Input ===
logger.info("\nWELCOME TO CRONUS AUTOMATED TRADING.")
if len(sys.argv) > 2:
    SYMBOL_A = sys.argv[1]
    SYMBOL_B = sys.argv[2]
else:
    logger.info("No command-line arguments provided. Using defaults...")
    SYMBOL_A = 'GBPUSD'
    SYMBOL_B = 'EURUSD'

# === Directories ===
optimal_zscore_dir = os.path.join(os.getcwd(), "optimal_zscores")
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(optimal_zscore_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# === Configuration ===
LOOKBACK_PERIOD = 5720
REGRESSION_PERIOD = 7020
DEFAULT_ENTRY_ZSCORE = 2.0  # For mean reversion
STOP_ZSCORE = 4.8
TAKE_PROFIT_ZSCORE = 0.0
RISK_PERCENT = 1.0
MIN_CORRELATION = 0.2
BYPASS_CORRELATION_CHECK = False
BYPASS_COINT_CHECK = True
MAGIC_NUMBER = 2024
RISK_REWARD_RATIO = 2.0
MAX_LOTS = 1.0
RESAMPLING = 2  # seconds
N_PATHS = 200
N_STEPS = 3600
MIN_COINT_PVALUE = 0.05
ZSCORE_MAX = 8.0
SIM_BALANCE = 10000  # Dummy for simulation sizing
plot_results = True

# === Global state ===
beta = 0.0
entry_zscore = DEFAULT_ENTRY_ZSCORE
spreads = np.zeros(LOOKBACK_PERIOD)
spread_idx = 0
last_optimize = time.time()
last_resample = time.time()

# === MT5 Initialization ===
def init_mt5():
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
    logger.info("MT5 initialized")
    if not mt5.symbol_select(SYMBOL_A, True) or not mt5.symbol_select(SYMBOL_B, True):
        logger.error(f"Cannot select symbols {SYMBOL_A}, {SYMBOL_B}")
        return False
    return True

# === Utility Functions ===
def custom_covariance(x, y):
    n = len(x)
    if n == 0 or len(y) != n:
        logger.warning("Arrays different sizes or empty")
        return 0
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / (n - 1)

def custom_correlation(x, y):
    cov = custom_covariance(x, y)
    std_x, std_y = np.std(x), np.std(y)
    if std_x == 0 or std_y == 0:
        logger.warning("Zero standard deviation in correlation")
        return 0
    return cov / (std_x * std_y)

def calculate_hedge_ratio():
    global beta
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(minutes=REGRESSION_PERIOD // 30)
    rates_a = mt5.copy_rates_range(SYMBOL_A, mt5.TIMEFRAME_M1, utc_from, utc_to)
    rates_b = mt5.copy_rates_range(SYMBOL_B, mt5.TIMEFRAME_M1, utc_from, utc_to)
    if rates_a is None or rates_b is None or len(rates_a) < 2 or len(rates_b) < 2:
        logger.error("Not enough M1 data for hedge ratio")
        return False
    closes_a = pd.Series([r['close'] for r in rates_a]).dropna()
    closes_b = pd.Series([r['close'] for r in rates_b]).dropna()
    min_len = min(len(closes_a), len(closes_b))
    if min_len < 2:
        logger.error("Not enough valid closing prices")
        return False
    closes_a = closes_a[-min_len:]
    closes_b = closes_b[-min_len:]
    cov = custom_covariance(closes_a.values, closes_b.values)
    var_b = np.var(closes_b.values)
    if var_b == 0 or np.isnan(var_b) or np.isnan(cov):
        logger.error("Invalid variance or covariance")
        return False
    beta = cov / var_b
    correlation = custom_correlation(closes_a.values, closes_b.values)
    if np.isnan(correlation):
        logger.error("Correlation is NaN")
        return False
    logger.info(f"Pair Correlation: {correlation:.4f}")
    logger.info(f"Calculated Beta: {beta:.4f}")
    return True

def get_spread_data():
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(seconds=REGRESSION_PERIOD * RESAMPLING)
    ticks_a = mt5.copy_ticks_range(SYMBOL_A, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    ticks_b = mt5.copy_ticks_range(SYMBOL_B, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    if ticks_a is None or ticks_b is None or len(ticks_a) < 2 or len(ticks_b) < 2:
        logger.warning("Insufficient tick data, falling back to M1")
        rates_a = mt5.copy_rates_range(SYMBOL_A, mt5.TIMEFRAME_M1, utc_from, utc_to)
        rates_b = mt5.copy_rates_range(SYMBOL_B, mt5.TIMEFRAME_M1, utc_from, utc_to)
        if rates_a is None or rates_b is None or len(rates_a) < 2 or len(rates_b) < 2:
            logger.error("Insufficient M1 data")
            return None, None, None, None
        df_a = pd.DataFrame(rates_a)[['time', 'close']].set_index('time')
        df_b = pd.DataFrame(rates_b)[['time', 'close']].set_index('time')
        df_a.index = pd.to_datetime(df_a.index, unit='s')
        df_b.index = pd.to_datetime(df_b.index, unit='s')
        return process_data(df_a, df_b)
    df_ticks_a = pd.DataFrame(ticks_a)[['time_msc', 'bid', 'ask']].dropna()
    df_ticks_b = pd.DataFrame(ticks_b)[['time_msc', 'bid', 'ask']].dropna()
    if df_ticks_a.empty or df_ticks_b.empty:
        logger.error("Tick data empty after dropping NaNs")
        return None, None, None, None
    df_ticks_a['time'] = pd.to_datetime(df_ticks_a['time_msc'], unit='ms')
    df_ticks_b['time'] = pd.to_datetime(df_ticks_b['time_msc'], unit='ms')
    df_ticks_a.set_index('time', inplace=True)
    df_ticks_b.set_index('time', inplace=True)
    df_resampled_a = df_ticks_a.resample(f'{RESAMPLING}s').last().ffill().dropna()
    df_resampled_b = df_ticks_b.resample(f'{RESAMPLING}s').last().ffill().dropna()
    df_a = pd.DataFrame({'close': (df_resampled_a['bid'] + df_resampled_a['ask']) / 2})
    df_b = pd.DataFrame({'close': (df_resampled_b['bid'] + df_resampled_b['ask']) / 2})
    if df_a.empty or df_b.empty:
        logger.error("Resampled data empty")
        return None, None, None, None
    return process_data(df_a, df_b)

def process_data(df_a, df_b):
    common_idx = df_a.index.intersection(df_b.index)
    df_a, df_b = df_a.loc[common_idx], df_b.loc[common_idx]
    if len(df_a) < REGRESSION_PERIOD * 0.8:
        logger.error("Insufficient overlapping data")
        return None, None, None, None
    coint_p, _, _ = coint(df_a['close'], df_b['close'])
    logger.info(f"Cointegration p-value: {coint_p:.4f}")
    if coint_p > MIN_COINT_PVALUE and not BYPASS_COINT_CHECK:
        logger.warning("Pair not cointegrated")
        return None, None, None, None
    pct_a = df_a['close'].pct_change().dropna().values
    pct_b = df_b['close'].pct_change().dropna().values
    min_len_pct = min(len(pct_a), len(pct_b))
    pct_a, pct_b = pct_a[:min_len_pct], pct_b[:min_len_pct]
    correlation = custom_correlation(pct_a, pct_b)
    if correlation < MIN_CORRELATION and not BYPASS_CORRELATION_CHECK:
        logger.warning(f"Correlation {correlation:.4f} below threshold {MIN_CORRELATION}")
        return None, None, None, None
    cov = custom_covariance(df_a['close'].values, df_b['close'].values)
    var_b = np.var(df_b['close'].values)
    local_beta = cov / var_b if var_b != 0 else 0
    recent_len = min(len(df_a), LOOKBACK_PERIOD + 1)
    recent_a = df_a['close'][-recent_len:]
    recent_b = df_b['close'][-recent_len:]
    spreads_local = recent_a - local_beta * recent_b
    mu, sigma = spreads_local[:-1].mean(), spreads_local[:-1].std()
    if sigma == 0 or np.isnan(sigma):
        logger.error("Invalid spread statistics")
        return None, None, None, None
    zscores = (spreads_local - mu) / sigma
    return zscores, mu, sigma, local_beta

# === Simulation Functions ===
def simulate_histogram(zscores, n_steps, n_paths):
    zscores = zscores[np.abs(zscores) < ZSCORE_MAX]
    if len(zscores) == 0:
        logger.warning("No valid Z-scores for simulation")
        return np.zeros((n_paths, n_steps))
    return np.random.choice(zscores, size=(n_paths, n_steps))

def simulate_portfolio(paths, mu, sigma, entry_zscores, stop_zscore, risk_reward_ratio):
    results = {}
    equity_curves = {z: [] for z in entry_zscores}
    pip_size = mt5.symbol_info(SYMBOL_A).point if mt5.symbol_info(SYMBOL_A) else 0.0001
    pip_value = mt5.symbol_info(SYMBOL_A).trade_tick_value if mt5.symbol_info(SYMBOL_A) else 10.0
    for entry_z in entry_zscores:
        pnls, total_trades = [], 0
        for path in paths:
            position, path_pnl, equity = 0, 0, [0.0] * len(path)
            for t in range(1, len(path)):
                z = path[t]
                if position == 0:
                    if z <= -entry_z:
                        position = 1
                        initial_spread = mu + z * sigma
                        delta = stop_zscore - entry_z  # Corrected for mean reversion
                        sl_z = -stop_zscore  # SL further from mean
                        tp_z = TAKE_PROFIT_ZSCORE  # Near mean (0.0)
                        total_trades += 1
                    elif z >= entry_z:
                        position = -1
                        initial_spread = mu + z * sigma
                        delta = stop_zscore - entry_z
                        sl_z = stop_zscore
                        tp_z = TAKE_PROFIT_ZSCORE
                        total_trades += 1
                else:
                    spread_now = mu + z * sigma
                    close_cond = (position == 1 and (z <= sl_z or z >= tp_z)) or \
                                 (position == -1 and (z >= sl_z or z <= tp_z))
                    if close_cond:
                        profit = (spread_now - initial_spread) * position / pip_size * pip_value * calculate_lots(sigma)  # Dynamic lots
                        path_pnl += profit
                        equity[t:] = [e + profit for e in equity[t:]]
                        position = 0
            if position != 0:
                spread_now = mu + path[-1] * sigma
                profit = (spread_now - initial_spread) * position / pip_size * pip_value * calculate_lots(sigma)
                path_pnl += profit
                equity[-1] += profit
            pnls.append(path_pnl)
            equity_curves[entry_z].append(equity)
        mean_pnl, std_pnl = np.mean(pnls), np.std(pnls)
        sharpe = mean_pnl / std_pnl if std_pnl != 0 else 0
        results[entry_z] = {'sharpe': sharpe, 'mean_pnl': mean_pnl, 'std_pnl': std_pnl, 'trades': total_trades}
    return results, equity_curves

def plot_results(paths, results, equity_curves, entry_zscores, zscores):
    # Sharpe Ratio
    plt.figure(figsize=(12, 6))
    plt.plot(entry_zscores, [results[z]['sharpe'] for z in entry_zscores], label='Sharpe Ratio', color='blue')
    plt.xlabel("Entry Z-Score")
    plt.ylabel("Sharpe Ratio")
    plt.title(f"Portfolio Effect by Entry Z-Score ({RESAMPLING}s)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{SYMBOL_A}-{SYMBOL_B}-portfolio_effect.png"))
    plt.close()

    # Z-Score Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(zscores, bins=40, alpha=0.5, label='Historical Z', density=True)
    plt.hist(paths.flatten(), bins=40, alpha=0.5, label='Simulated Z', density=True)
    plt.legend()
    plt.title("Historical vs Simulated Z-Score Distribution")
    plt.xlabel("Z-Score")
    plt.ylabel("Density")
    plt.grid()
    plt.savefig(os.path.join(results_dir, f"{SYMBOL_A}-{SYMBOL_B}-zscore_histogram.png"))
    plt.close()

    # Equity Curves
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(entry_zscores)))
    for idx, z in enumerate(entry_zscores):
        if equity_curves[z]:
            avg_equity = np.mean(equity_curves[z], axis=0)
            plt.plot(range(N_STEPS), avg_equity, label=f'Z={z:.1f}', color=colors[idx], alpha=0.7)
    plt.xlabel(f"Steps ({RESAMPLING}s)")
    plt.ylabel("Equity (USD)")
    plt.title("Average Equity Curves by Entry Z-Score")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, f"{SYMBOL_A}-{SYMBOL_B}-equity_curves.png"))
    plt.close()

# === Position Management Functions ===
def get_position_direction():
    positions = mt5.positions_get(symbol=SYMBOL_A) or []
    positions += mt5.positions_get(symbol=SYMBOL_B) or []
    pos_a, pos_b = None, None
    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue
        if pos.symbol == SYMBOL_A:
            pos_a = pos
        elif pos.symbol == SYMBOL_B:
            pos_b = pos
    if not pos_a or not pos_b:
        return 0
    if pos_a.type == mt5.POSITION_TYPE_BUY and pos_b.type == mt5.POSITION_TYPE_SELL:
        return 1  # Long spread
    if pos_a.type == mt5.POSITION_TYPE_SELL and pos_b.type == mt5.POSITION_TYPE_BUY:
        return -1  # Short spread
    return 0

def calculate_lots(sigma):
    account_info = mt5.account_info()
    if not account_info:
        logger.error("Failed to get account info")
        return LOT_SIZE
    account_balance = account_info.balance if account_info else SIM_BALANCE  # Use sim balance if MT5 not available
    account_risk = account_balance * RISK_PERCENT / 100
    pip_size = mt5.symbol_info(SYMBOL_A).point if mt5.symbol_info(SYMBOL_A) else 0.0001
    pip_value = mt5.symbol_info(SYMBOL_A).trade_tick_value if mt5.symbol_info(SYMBOL_A) else 10.0
    lots = account_risk / (sigma / pip_size * pip_value) if sigma != 0 else LOT_SIZE
    symbol_info = mt5.symbol_info(SYMBOL_A)
    if symbol_info:
        lot_step = symbol_info.volume_step
        min_lot = symbol_info.volume_min
        max_lot = min(symbol_info.volume_max, MAX_LOTS)
        lots = max(min_lot, min(max_lot, np.floor(lots / lot_step) * lot_step))  # Floor for conservative risk
    return lots

def open_long_spread(lots_a, lots_b, z):
    symbol_info_a = mt5.symbol_info(SYMBOL_A)
    symbol_info_b = mt5.symbol_info(SYMBOL_B)
    if not symbol_info_a or not symbol_info_b:
        logger.error(f"Cannot get symbol info for {SYMBOL_A} or {SYMBOL_B}")
        return
    lots_b = max(symbol_info_b.volume_min, min(min(symbol_info_b.volume_max, MAX_LOTS),
                                               np.floor(lots_b / symbol_info_b.volume_step) * symbol_info_b.volume_step))
    request_a = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL_A,
        "volume": lots_a,
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(SYMBOL_A).ask,
        "magic": MAGIC_NUMBER,
        "comment": "Long Spread",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    request_b = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL_B,
        "volume": lots_b,
        "type": mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(SYMBOL_B).bid,
        "magic": MAGIC_NUMBER,
        "comment": "Long Spread",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result_a = mt5.order_send(request_a)
    result_b = mt5.order_send(request_b)
    if result_a.retcode == mt5.TRADE_RETCODE_DONE and result_b.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Opened LONG spread Z={z:.2f}, Lots A={lots_a:.2f}, Lots B={lots_b:.2f}")
    else:
        logger.error(f"Failed to open long spread: A={result_a.comment if result_a else 'None'}, B={result_b.comment if result_b else 'None'}")

def open_short_spread(lots_a, lots_b, z):
    symbol_info_a = mt5.symbol_info(SYMBOL_A)
    symbol_info_b = mt5.symbol_info(SYMBOL_B)
    if not symbol_info_a or not symbol_info_b:
        logger.error(f"Cannot get symbol info for {SYMBOL_A} or {SYMBOL_B}")
        return
    lots_b = max(symbol_info_b.volume_min, min(min(symbol_info_b.volume_max, MAX_LOTS),
                                               np.floor(lots_b / symbol_info_b.volume_step) * symbol_info_b.volume_step))
    request_a = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL_A,
        "volume": lots_a,
        "type": mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(SYMBOL_A).bid,
        "magic": MAGIC_NUMBER,
        "comment": "Short Spread",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    request_b = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL_B,
        "volume": lots_b,
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(SYMBOL_B).ask,
        "magic": MAGIC_NUMBER,
        "comment": "Short Spread",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result_a = mt5.order_send(request_a)
    result_b = mt5.order_send(request_b)
    if result_a.retcode == mt5.TRADE_RETCODE_DONE and result_b.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Opened SHORT spread Z={z:.2f}, Lots A={lots_a:.2f}, Lots B={lots_b:.2f}")
    else:
        logger.error(f"Failed to open short spread: A={result_a.comment if result_a else 'None'}, B={result_b.comment if result_b else 'None'}")

def calculate_pair_profit():
    positions = mt5.positions_get(symbol=SYMBOL_A) or []
    positions += mt5.positions_get(symbol=SYMBOL_B) or []
    profit = 0.0
    for pos in positions:
        if pos.magic == MAGIC_NUMBER:
            profit += pos.profit
    return profit

def check_close_conditions(direction, z):
    if direction == 0:
        return False
    if direction == 1:
        return z >= TAKE_PROFIT_ZSCORE or z <= -STOP_ZSCORE
    if direction == -1:
        return z <= TAKE_PROFIT_ZSCORE or z >= STOP_ZSCORE
    return False

def close_all_positions():
    positions = mt5.positions_get(symbol=SYMBOL_A) or []
    positions += mt5.positions_get(symbol=SYMBOL_B) or []
    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
            "magic": MAGIC_NUMBER,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Closed position {pos.ticket} for {pos.symbol}, profit={pos.profit:.2f}")
        else:
            logger.error(f"Failed to close position {pos.ticket}: {result.comment if result else 'None'}")

def is_market_open():
    sym_a, sym_b = mt5.symbol_info(SYMBOL_A), mt5.symbol_info(SYMBOL_B)
    if sym_a is None or sym_b is None:
        logger.error(f"Cannot get symbol info for {SYMBOL_A} or {SYMBOL_B}")
        return False
    return sym_a.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED and sym_b.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED

# === Update Entry Z-Score ===
def update_entry_zscore():
    global entry_zscore
    zscores, mu, sigma, _ = get_spread_data()
    if zscores is None:
        return
    paths = simulate_histogram(zscores, N_STEPS, N_PATHS)
    entry_zscores = np.arange(0.5, ZSCORE_MAX + 0.5, 0.5)
    results, equity_curves = simulate_portfolio(paths, mu, sigma, entry_zscores, STOP_ZSCORE, RISK_REWARD_RATIO)

    #if plot results is set to true...
    if plot_results:
        plot_results(paths, results, equity_curves, entry_zscores, zscores)

    #continue with the rest of results processing logic
    max_sharpe, optimal_z = -float('inf'), DEFAULT_ENTRY_ZSCORE
    for z in results:
        if results[z]['mean_pnl'] > 0 and results[z]['sharpe'] > max_sharpe:
            max_sharpe = results[z]['sharpe']
            optimal_z = z
    entry_zscore = optimal_z
    logger.info(f"Updated Entry Z-Score: {entry_zscore:.2f}")
    with open(os.path.join(optimal_zscore_dir, f"{SYMBOL_A}-{SYMBOL_B}-optimal_zscore.txt"), "w") as f:
        f.write(str(entry_zscore))

# === Signal Handler for Graceful Exit ===
def signal_handler(sig, frame):
    logger.info("\nShutting down CRONUS...")
    mt5.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Main Loop ===
if __name__ == "__main__":
    if not init_mt5():
        sys.exit(1)
    if not calculate_hedge_ratio():
        mt5.shutdown()
        sys.exit(1)
    try:
        while True:
            if not mt5.initialize():  # Reconnect if needed
                logger.error("MT5 reconnection failed")
                time.sleep(10)
                continue
            now = time.time()
            if now - last_optimize >= 60:   #update zscores after this seconds
                update_entry_zscore()
                last_optimize = now
            if not is_market_open():
                logger.info("Market closed, waiting...")
                time.sleep(10)
                continue
            if now - last_resample >= RESAMPLING:   #fetch tick data based on resampling period
                last_resample = now
                tick_a, tick_b = mt5.symbol_info_tick(SYMBOL_A), mt5.symbol_info_tick(SYMBOL_B)
                if tick_a is None or tick_b is None:
                    logger.error("Failed to get tick data")
                    continue
                spread_now = ((tick_a.bid + tick_a.ask) / 2) - beta * ((tick_b.bid + tick_b.ask) / 2)
                if spread_idx < LOOKBACK_PERIOD:
                    spreads[spread_idx] = spread_now
                    spread_idx += 1
                else:
                    spreads = np.roll(spreads, -1)
                    spreads[-1] = spread_now
                if spread_idx >= LOOKBACK_PERIOD:
                    mu = np.mean(spreads)
                    sigma = np.std(spreads)
                    if sigma == 0:
                        logger.warning("Zero spread standard deviation")
                        continue
                    z = (spread_now - mu) / sigma
                    logger.info(f"[{datetime.now()}] Z-Score: {z:.4f}")
                    direction = get_position_direction()
                    if direction == 0:
                        if z <= -entry_zscore:
                            lots_a = calculate_lots(sigma)
                            lots_b = beta * lots_a
                            open_long_spread(lots_a, lots_b, z)
                        elif z >= entry_zscore:
                            lots_a = calculate_lots(sigma)
                            lots_b = beta * lots_a
                            open_short_spread(lots_a, lots_b, z)
                    else:
                        if check_close_conditions(direction, z):
                            close_all_positions()
            time.sleep(0.1) #sleep for this seconds to reduce CPU load
    except KeyboardInterrupt:
        signal_handler(None, None)