# run_optimize.py
import os
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fetch_mt5 import mt5_initialize, fetch_bars
from preprocess import align_series, compute_beta
from optimizer import run_optimizer, build_walkforward_windows, evaluate_params_on_splits
from export_set import export_setfile
from utils import sharpe_from_equity
from simulator import backtest

def plot_equity(eq, outdir, label="Final Equity Curve", fname="final_equity.png"):
    plt.figure(figsize=(12, 6))
    eq.plot(title=label, linewidth=1.2)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fn = os.path.join(outdir, fname)
    plt.savefig(fn)
    plt.close()
    print("Saved equity curve plot:", fn)

def plot_sharpe_distribution(results_file, outdir):
    df = pd.read_csv(results_file)
    if "score" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(df["score"], bins=30, kde=True)
    plt.title("Sharpe Ratio Distribution Across Trials")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    fn = os.path.join(outdir, "sharpe_hist.png")
    plt.savefig(fn)
    plt.close()
    print("Saved Sharpe distribution plot:", fn)

def plot_heatmap(results_file, outdir):
    df = pd.read_csv(results_file)
    if not {"entry_z", "stop_z", "score"}.issubset(df.columns):
        return
    pivot = df.pivot_table(index="stop_z", columns="entry_z", values="score", aggfunc="mean")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="coolwarm", annot=False, cbar_kws={"label": "Sharpe"})
    plt.title("Sharpe Ratio Heatmap (entry_z vs stop_z)")
    plt.xlabel("Entry Z-Score")
    plt.ylabel("Stop Z-Score")
    plt.tight_layout()
    fn = os.path.join(outdir, "heatmap_entry_stop.png")
    plt.savefig(fn)
    plt.close()
    print("Saved heatmap plot:", fn)

def main(config_path="config.yaml"):
    # Load config
    cfg = yaml.safe_load(open(config_path))
    data_cfg = cfg['data']
    mt5_cfg = cfg['mt5']
    outdir = cfg['output']['save_dir']
    os.makedirs(outdir, exist_ok=True)

    # Load or fetch data
    data_dir = "data"
    fA = os.path.join(data_dir, f"{data_cfg['symbol_a']}_{data_cfg['timeframe']}.parquet")
    fB = os.path.join(data_dir, f"{data_cfg['symbol_b']}_{data_cfg['timeframe']}.parquet")
    if not (os.path.exists(fA) and os.path.exists(fB)):
        print("Local data not found. Fetching from MT5...")
        mt5_initialize(mt5_cfg)
        dfA = fetch_bars(data_cfg['symbol_a'], timeframe=data_cfg['timeframe'],
                         start=mt5_cfg['from'], end=mt5_cfg['to'])
        dfB = fetch_bars(data_cfg['symbol_b'], timeframe=data_cfg['timeframe'],
                         start=mt5_cfg['from'], end=mt5_cfg['to'])
        os.makedirs(data_dir, exist_ok=True)
        dfA.to_parquet(fA)
        dfB.to_parquet(fB)
    else:
        dfA = pd.read_parquet(fA)
        dfB = pd.read_parquet(fB)

    print("Aligning series...")
    df = align_series(dfA, dfB)

    print("Computing static beta (last regression period)...")
    beta = compute_beta(df, regression_period=cfg['strategy']['regression_period'])
    print("Beta:", beta)

    print("Running optimizer...")
    best_params, res_file = run_optimizer(df, cfg, beta, outdir)
    print("Optimization results saved to:", res_file)
    print("Best params:", best_params)

    # Build final config
    final_cfg = dict(cfg)
    final_cfg['strategy'] = dict(cfg['strategy'])
    for k, v in best_params.items():
        final_cfg['strategy'][k] = v

    # Walk-forward settings
    n_splits = int(cfg['optimizer'].get('walk_forward_splits', 0))
    train_days = int(cfg['optimizer'].get('train_days') or 0)
    test_days = int(cfg['optimizer'].get('test_days') or 0)

    if n_splits > 0:
        print(f"Evaluating best params with Walk-Forward ({n_splits} splits)...")
        wf_windows = build_walkforward_windows(df.index, train_days, test_days, n_splits)
        mean_sharpe, split_sharpes = evaluate_params_on_splits(best_params, df, beta, final_cfg, wf_windows)

        print("Sharpe per split:", split_sharpes)
        print("Average Test Sharpe:", mean_sharpe)

        # Save per-split equity curves
        for idx, (train_start, train_end, test_start, test_end) in enumerate(wf_windows):
            df_test = df.loc[(df.index >= test_start) & (df.index <= test_end)]
            eq, trades = backtest(df_test, beta, final_cfg)
            eq.to_csv(os.path.join(outdir, f"equity_split{idx}.csv"))
            trades.to_csv(os.path.join(outdir, f"trades_split{idx}.csv"), index=False)
            plot_equity(eq, outdir, label=f"Equity Curve Split {idx}", fname=f"equity_split{idx}.png")
    else:
        print("Evaluating best params on full dataset...")
        eq, trades = backtest(df, beta, final_cfg)
        eq.to_csv(os.path.join(outdir, "final_equity.csv"))
        trades.to_csv(os.path.join(outdir, "final_trades.csv"), index=False)
        final_sharpe = sharpe_from_equity(eq, timeframe=cfg['data']['timeframe'])
        print("Final Sharpe ratio:", final_sharpe)
        plot_equity(eq, outdir)

    # Extra plots
    plot_sharpe_distribution(res_file, outdir)
    if cfg['optimizer']['method'] == "grid":
        plot_heatmap(res_file, outdir)

    # Export .set file
    if cfg['output'].get('export_set_file', True):
        params = {
            "SymbolA": cfg['data']['symbol_a'],
            "SymbolB": cfg['data']['symbol_b'],
            "RegressionPeriod": cfg['strategy']['regression_period'],
            "LookbackPeriod": final_cfg['strategy']['lookback_period'],
            "EntryZScore": final_cfg['strategy']['entry_z'],
            "StopZScore": final_cfg['strategy']['stop_z'],
            "RiskPercent": final_cfg['strategy']['risk_percent'],
            "MinCorrelation": cfg['strategy']['min_correlation'],
            "RiskRewardRatio": cfg['strategy']['tp_multiple'],
            "StopLossPercent": cfg['strategy']['stoploss_percent'],
            "MaxLots": cfg['strategy']['max_lots']
        }
        setfile = export_setfile(params, filename=os.path.join(outdir, "REHOBOAM_optimized.set"))
        print("Exported .set file:", setfile)

    print("✅ Done. Results saved to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
