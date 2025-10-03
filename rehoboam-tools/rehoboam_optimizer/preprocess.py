# preprocess.py
import pandas as pd
import numpy as np
from scipy import stats

def load_parquet(path):
    return pd.read_parquet(path)

def align_series(dfA, dfB):
    # Align by index (time); inner join to ensure same timestamps
    df = pd.concat([dfA['close'], dfB['close']], axis=1, keys=['A', 'B']).dropna()
    df.columns = ['closeA', 'closeB']
    return df

def compute_beta(df, regression_period=252):
    # Use last regression_period rows to compute Beta via least squares on prices
    n = min(len(df), regression_period)
    y = df['closeA'].values[-n:]
    x = df['closeB'].values[-n:]
    X = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    beta = m
    return beta

def rolling_beta_series(df, regression_period=252):
    # Rolling linear regression slope (beta) using numpy/scipy
    betas = pd.Series(index=df.index, dtype=float)
    for i in range(regression_period, len(df)):
        window = df.iloc[i-regression_period:i]
        x = window['closeB'].values
        y = window['closeA'].values
        m, c = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
        betas.iloc[i] = m
    return betas.fillna(method='ffill')

def compute_spread(df, beta):
    return df['closeA'] - beta * df['closeB']
