# simulator.py
import numpy as np
import pandas as pd
from numba import njit
import math

def compute_rolling_stats(spread, lookback):
    # returns rolling mean and std (aligned: at index i use previous lookback values)
    mean = pd.Series(spread).rolling(window=lookback, min_periods=lookback).mean().shift(1)
    std = pd.Series(spread).rolling(window=lookback, min_periods=lookback).std(ddof=0).shift(1)
    return mean.values, std.values

@njit
def simulate_loop(spread_arr, mu_arr, sigma_arr, entry_z, stop_z, tp_z,
                  tp_type_int, sl_type_int, risk_percent, initial_balance,
                  latency_ms, slippage_std, commission, samples_per_minute):
    """
    tp_type_int: 0 -> TP_Multiple, 1 -> TP_ZScore
    sl_type_int: 0 -> SL_ZScore, 1 -> SL_Percent (percent not used in this simplified numeric simulation)
    We model PnL in account currency via risk sizing: size = risk_amount / stop_distance
    Returns equity series (length N), trades list: [entry_index, exit_index, pnl]
    """
    N = len(spread_arr)
    equity = np.empty(N)
    equity[0] = initial_balance
    position = 0  # 0 none, 1 long spread, -1 short spread
    entry_spread = 0.0
    entry_idx = -1
    entry_equity = initial_balance
    trades = []
    cash = initial_balance
    # Latency factor translate ms into fraction of a minute
    latency_frac = latency_ms / 60000.0
    for i in range(1, N):
        equity[i] = cash
        mu = mu_arr[i]
        sigma = sigma_arr[i]
        if np.isnan(mu) or np.isnan(sigma) or sigma <= 0.0:
            continue
        z = (spread_arr[i] - mu) / sigma
        # If no position, check entry
        if position == 0:
            if z <= -entry_z:
                # open long spread: buy A, sell B -> profit when spread rises back to mean
                # simulate slippage as normal with std = slippage_std * sigma
                slippage = np.random.normal(0.0, slippage_std * sigma)
                entry_spread = spread_arr[i] + slippage
                entry_idx = i
                entry_equity = cash
                # compute stop distance in spread units
                stop_spread = mu - stop_z * sigma if sl_type_int == 0 else entry_spread - (stop_z * sigma)
                # stop distance positive
                stop_distance = abs(entry_spread - stop_spread)
                if stop_distance <= 0:
                    stop_distance = 1e-9
                position = 1
            elif z >= entry_z:
                # open short spread (sell A, buy B)
                slippage = np.random.normal(0.0, slippage_std * sigma)
                entry_spread = spread_arr[i] + slippage
                entry_idx = i
                entry_equity = cash
                stop_spread = mu + stop_z * sigma if sl_type_int == 0 else entry_spread + (stop_z * sigma)
                stop_distance = abs(entry_spread - stop_spread)
                if stop_distance <= 0:
                    stop_distance = 1e-9
                position = -1
            else:
                continue
        else:
            # Position open: check exit on each subsequent bar
            # simulate slippage/latency effect on observed spread at i
            observed_spread = spread_arr[i]  # we use current spread for checking
            # compute z at time i
            z_i = (observed_spread - mu) / sigma
            # Determine stop/TP z thresholds
            if position == 1:
                if sl_type_int == 0:
                    sl_z = (entry_spread - (mu - stop_z * sigma)) / sigma  # not exact; use simpler condition below
                    # For long, stop occurs if z <= mu_z - (stop_z?) simpler: if z <= ( -stop_z + ???)
                    # Simpler rule: if z <= -stop_z then stop; if tp_type is ZScore check z >= tp_z
                # Apply direct checks:
                stop_hit = (z_i <= -stop_z)
                if tp_type_int == 0:
                    # tp by multiple: target z = entry_z + (tp_multiple*(stop_z - entry_z)) -> map to z check
                    # But simpler: treat tp as reaching smaller |z|: z >= 0 or z >= tp_z
                    tp_hit = (z_i >= tp_z)
                else:
                    tp_hit = (z_i >= tp_z)
                if stop_hit or tp_hit:
                    # compute pnl using risk sizing: size = risk_amount / stop_distance
                    # estimate stop_distance in spread units:
                    stop_spread = mu - stop_z * sigma
                    stop_distance = abs(entry_spread - stop_spread)
                    if stop_distance <= 0:
                        stop_distance = 1e-9
                    risk_amount = (risk_percent / 100.0) * entry_equity
                    size = risk_amount / stop_distance
                    # realized pnl = (exit_spread - entry_spread) * size - commission
                    exit_spread = observed_spread + np.random.normal(0.0, slippage_std * sigma)
                    pnl = (exit_spread - entry_spread) * size - commission
                    cash = cash + pnl
                    trades.append((entry_idx, i, pnl))
                    position = 0
            elif position == -1:
                stop_hit = (z_i >= stop_z)
                if tp_type_int == 0:
                    tp_hit = (z_i <= tp_z)
                else:
                    tp_hit = (z_i <= tp_z)
                if stop_hit or tp_hit:
                    stop_spread = mu + stop_z * sigma
                    stop_distance = abs(entry_spread - stop_spread)
                    if stop_distance <= 0:
                        stop_distance = 1e-9
                    risk_amount = (risk_percent / 100.0) * entry_equity
                    size = risk_amount / stop_distance
                    exit_spread = observed_spread + np.random.normal(0.0, slippage_std * sigma)
                    # for short, pnl = (entry - exit) * size
                    pnl = (entry_spread - exit_spread) * size - commission
                    cash = cash + pnl
                    trades.append((entry_idx, i, pnl))
                    position = 0
    return equity, trades

def backtest(df, beta, cfg):
    # df must have columns closeA, closeB
    df = df.copy()
    df['spread'] = df['closeA'] - beta * df['closeB']
    lookback = int(cfg['strategy']['lookback_period'])
    mu_arr, sigma_arr = compute_rolling_stats(df['spread'].values, lookback)
    entry_z = float(cfg['strategy'].get('entry_z') or cfg['strategy']['entry_z'])
    stop_z = float(cfg['strategy'].get('stop_z') or cfg['strategy']['stop_z'])
    tp_z = float(cfg['strategy'].get('takeprofit_z') or cfg['strategy']['takeprofit_z'])
    tp_type = 0 if cfg['strategy']['tp_type'] == "TP_Multiple" else 1
    sl_type = 0 if cfg['strategy']['sl_type'] == "SL_ZScore" else 1
    risk_percent = float(cfg['strategy'].get('risk_percent', 1.0))
    initial_balance = float(cfg.get('initial_balance', 10000.0))
    latency_ms = float(cfg['execution'].get('latency_ms', 221))
    slippage_std = float(cfg['execution'].get('slippage_std_points', 0.25))
    commission = float(cfg['execution'].get('commission_per_trade', 0.0))
    samples_per_minute = 1  # we're using M1 bars
    equity, trades = simulate_loop(df['spread'].values.astype(np.float64),
                                   mu_arr.astype(np.float64),
                                   sigma_arr.astype(np.float64),
                                   entry_z, stop_z, tp_z,
                                   tp_type, sl_type, risk_percent,
                                   initial_balance, latency_ms, slippage_std,
                                   commission, samples_per_minute)
    eq_series = pd.Series(equity, index=df.index)
    trades_df = pd.DataFrame(trades, columns=['entry_idx', 'exit_idx', 'pnl'])
    return eq_series, trades_df
