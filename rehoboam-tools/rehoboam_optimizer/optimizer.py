# optimizer.py
import os
import math
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import optuna
from tqdm import tqdm

from simulator import backtest
from utils import sharpe_from_equity

# ---------------------------
# Floating range
# ---------------------------
def _frange(start, stop, step):
    vals = []
    v = float(start)
    while v <= stop + 1e-12:
        vals.append(round(v, 12))
        v += step
    return vals

# ---------------------------
# Grid expansion
# ---------------------------
def expand_grid_params(grid_params):
    combos = []
    keys = []
    values = []
    for k, spec in grid_params.items():
        keys.append(k)
        if isinstance(spec, dict) and {"start","stop","step"} <= spec.keys():
            vals = _frange(float(spec["start"]), float(spec["stop"]), float(spec["step"]))
        elif isinstance(spec, (list,tuple)):
            vals = list(spec)
        else:
            vals = [spec]
        values.append(vals)
    for prod in itertools.product(*values):
        combos.append(dict(zip(keys, prod)))
    return combos

# ---------------------------
# Walk-forward windows
# ---------------------------
def build_walkforward_windows(df_index, train_days, test_days, n_splits):
    if n_splits <= 0:
        return []
    start_ts = df_index[0]
    end_ts = df_index[-1]
    windows = []
    for i in range(n_splits):
        train_start = start_ts + pd.Timedelta(days=i*test_days)
        train_end   = train_start + pd.Timedelta(days=train_days) - pd.Timedelta(seconds=1)
        test_start  = train_end + pd.Timedelta(seconds=1)
        test_end    = test_start + pd.Timedelta(days=test_days) - pd.Timedelta(seconds=1)
        if test_end > end_ts:
            break
        windows.append((train_start, train_end, test_start, test_end))
    return windows

# ---------------------------
# Evaluate params (WFO-aware)
# ---------------------------
def evaluate_params(params, df, beta, cfg, wf_windows):
    per_split_sharpes = []
    equities = []

    # if no WFO → full dataset backtest
    if not wf_windows:
        cfg_local = dict(cfg)
        cfg_local["strategy"] = dict(cfg["strategy"])
        cfg_local["strategy"].update(params)
        eq, _ = backtest(df, beta, cfg_local)
        sh = sharpe_from_equity(eq, timeframe=cfg["data"]["timeframe"])
        return sh, [sh], sh, eq

    # run across splits
    for (train_start, train_end, test_start, test_end) in wf_windows:
        df_test = df.loc[(df.index >= test_start) & (df.index <= test_end)]
        if df_test.empty:
            per_split_sharpes.append(-999.0)
            continue
        cfg_local = dict(cfg)
        cfg_local["strategy"] = dict(cfg["strategy"])
        cfg_local["strategy"].update(params)
        try:
            eq, _ = backtest(df_test, beta, cfg_local)
            sh = sharpe_from_equity(eq, timeframe=cfg["data"]["timeframe"])
            per_split_sharpes.append(sh)
            equities.append(eq)
        except Exception:
            per_split_sharpes.append(-999.0)

    if not equities:
        return -999.0, per_split_sharpes, -999.0, pd.Series(dtype=float)

    # compute average split sharpe
    avg_split_sharpe = float(np.mean([s for s in per_split_sharpes if math.isfinite(s)]))

    # stitch equities
    combined = []
    running_end = None
    for eq in equities:
        eq = eq.copy()
        if running_end is None:
            combined.append(eq)
            running_end = eq.iloc[-1]
        else:
            shift = running_end - eq.iloc[0]
            eq = eq + shift
            combined.append(eq)
            running_end = eq.iloc[-1]
    combined_eq = pd.concat(combined).sort_index()
    combined_sharpe = sharpe_from_equity(combined_eq, timeframe=cfg["data"]["timeframe"])

    return combined_sharpe, per_split_sharpes, avg_split_sharpe, combined_eq

#-----------------
# dummy wrapper
#-------------
def evaluate_params_on_splits(params, df, beta, cfg, wf_windows):
    # Keep for compatibility – forward to evaluate_params and return tuple
    comb_sh, splits, avg_sh, eq = evaluate_params(params, df, beta, cfg, wf_windows)
    return avg_sh, splits

# ---------------------------
# Worker for grid search
# ---------------------------
def _grid_worker(args):
    params, df_pickle_path, beta, cfg, wf_windows = args
    df = pd.read_pickle(df_pickle_path)
    comb_sh, splits, avg_sh, _ = evaluate_params(params, df, beta, cfg, wf_windows)
    return params, comb_sh, splits, avg_sh

# ---------------------------
# Grid search
# ---------------------------
def grid_search(df, cfg, beta, outdir):
    grid_params = cfg.get("optimize", {})
    defaults = cfg["strategy"]

    combos = expand_grid_params(grid_params)
    print(f"Grid has {len(combos)} combinations.")

    wf_windows = build_walkforward_windows(
        df.index,
        cfg["optimizer"].get("train_days",0),
        cfg["optimizer"].get("test_days",0),
        cfg["optimizer"].get("walk_forward_splits",0)
    )

    os.makedirs(outdir, exist_ok=True)
    df_pickle_path = os.path.join(outdir,"df.pkl")
    df.to_pickle(df_pickle_path)

    args_list = []
    for params in combos:
        merged = dict(defaults)
        merged.update(params)
        args_list.append((merged, df_pickle_path, beta, cfg, wf_windows))

    results = []
    with ProcessPoolExecutor(max_workers=cfg["optimizer"].get("n_jobs",1)) as exe:
        futures = {exe.submit(_grid_worker,a):a[0] for a in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Grid eval"):
            try:
                params, comb_sh, splits, avg_sh = fut.result()
                results.append((params, comb_sh, splits, avg_sh))
            except Exception:
                results.append((futures[fut], -999.0, [], -999.0))

    # build results dataframe
    rows = []
    for params, comb_sh, splits, avg_sh in results:
        row = dict(params)
        row["avg_split_sharpe"] = avg_sh
        row["combined_sharpe"] = comb_sh
        for i, s in enumerate(splits):
            row[f"split_{i}_sharpe"] = s
        rows.append(row)

    res_df = pd.DataFrame(rows)
    res_file = os.path.join(outdir,"grid_results.csv")
    res_df.to_csv(res_file,index=False)

    # pick best by combined_sharpe
    best_idx = res_df["combined_sharpe"].idxmax()
    best_params = res_df.loc[best_idx].to_dict()
    # filter only optimized keys
    final_params = {k: best_params[k] for k in grid_params.keys()}
    return final_params, res_file

# ---------------------------
# Optuna search
# ---------------------------
def optuna_search(df, cfg, beta, outdir):
    wf_windows = build_walkforward_windows(
        df.index,
        cfg["optimizer"].get("train_days",0),
        cfg["optimizer"].get("test_days",0),
        cfg["optimizer"].get("walk_forward_splits",0)
    )

    optimize_params = cfg.get("optimize", {})
    defaults = cfg["strategy"]

    def objective(trial):
        params = {}
        for k, spec in optimize_params.items():
            if isinstance(spec, dict) and {"start","stop"} <= spec.keys():
                if spec.get("type","float")=="int":
                    params[k] = trial.suggest_int(k,int(spec["start"]),int(spec["stop"]))
                else:
                    params[k] = trial.suggest_float(k,float(spec["start"]),float(spec["stop"]))
        merged = dict(defaults)
        merged.update(params)
        comb_sh, splits, avg_sh, _ = evaluate_params(merged, df, beta, cfg, wf_windows)
        return comb_sh

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["optimizer"].get("seed",42)))
    study.optimize(objective, n_trials=cfg["optimizer"].get("trials",100))

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(outdir,"optuna_trials.csv"),index=False)

    best = study.best_trial.params
    return best, os.path.join(outdir,"optuna_trials.csv")

# ---------------------------
# Main runner
# ---------------------------
def run_optimizer(df, cfg, beta, outdir):
    method = cfg["optimizer"].get("method","grid").lower()
    if method=="grid":
        return grid_search(df, cfg, beta, outdir)
    elif method=="optuna":
        return optuna_search(df, cfg, beta, outdir)
    else:
        raise ValueError("optimizer.method must be 'grid' or 'optuna'")
