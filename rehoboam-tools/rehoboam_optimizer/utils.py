# utils.py
import numpy as np
import pandas as pd

def sharpe_from_equity(eq_series, timeframe='M1'):
    # eq_series is pandas Series of equity over time (index timestamps)
    # compute periodic returns
    returns = eq_series.pct_change().fillna(0.0)
    mean_r = returns.mean()
    std_r = returns.std(ddof=0)
    if std_r == 0:
        return 0.0
    # determine periods_per_year based on timeframe
    if timeframe == 'M1':
        periods_per_day = 24 * 60
    elif timeframe == 'H1':
        periods_per_day = 24
    elif timeframe == 'D1':
        periods_per_day = 1
    else:
        periods_per_day = 24 * 60
    periods_per_year = periods_per_day * 252
    sharpe = (mean_r / std_r) * (periods_per_year ** 0.5)
    return float(sharpe)
