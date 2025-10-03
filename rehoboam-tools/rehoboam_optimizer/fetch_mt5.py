# fetch_mt5.py
import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import yaml

def mt5_initialize(cfg):
    server = cfg.get("server") or None
    login = int(cfg.get("login") or 0)
    password = cfg.get("password") or None
    port = cfg.get("port") or None
    # Usually local terminal is fine; mt5.initialize() auto-detects terminal.
    if server:
        ok = mt5.initialize(server, login=login, password=password, port=port)
    else:
        ok = mt5.initialize()
    if not ok:
        raise RuntimeError("MT5 initialize failed: " + str(mt5.last_error()))
    return True

def fetch_bars(symbol, timeframe="M1", start=None, end=None):
    # timeframe is "M1" etc. We'll map to mt5 constant
    tf_map = {"M1": mt5.TIMEFRAME_M1, "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}
    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M1)
    utc_from = int(pd.to_datetime(start).timestamp())
    utc_to = int(pd.to_datetime(end).timestamp())
    rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {symbol} in range {start} to {end}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def main(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    mt5cfg = cfg['mt5']
    data_cfg = cfg['data']
    outdir = "data"
    os.makedirs(outdir, exist_ok=True)
    mt5_initialize(mt5cfg)
    s = data_cfg['symbol_a']
    t = data_cfg['symbol_b']
    start = mt5cfg.get('from')
    end = mt5cfg.get('to')
    print("Fetching", s)
    dfA = fetch_bars(s, timeframe=data_cfg['timeframe'], start=start, end=end)
    fnA = os.path.join(outdir, f"{s}_{data_cfg['timeframe']}.parquet")
    dfA.to_parquet(fnA)
    print("Saved", fnA)
    print("Fetching", t)
    dfB = fetch_bars(t, timeframe=data_cfg['timeframe'], start=start, end=end)
    fnB = os.path.join(outdir, f"{t}_{data_cfg['timeframe']}.parquet")
    dfB.to_parquet(fnB)
    print("Saved", fnB)
    mt5.shutdown()

if __name__ == "__main__":
    main()
