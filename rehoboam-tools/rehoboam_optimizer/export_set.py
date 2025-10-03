def export_setfile(params, filename="REHOBOAM_optimized.set"):
    keys_order = [
        "SymbolA", "SymbolB", "Timeframe",
        "LookbackPeriod", "RegressionPeriod",
        "EntryZScore", "StopZScore", "TakeProfitZScore",
        "RiskPercent", "MinCorrelation", "BypassCorrelationCheck",
        "MagicNumber", "RiskRewardRatio",
        "SL_Type", "TP_Type", "StopLossPercent", "MaxLots"
    ]
    with open(filename, "w") as f:
        for k in keys_order:
            if k not in params:
                continue
            v = params[k]
            if isinstance(v, bool):
                v = str(v).lower()  # MT5 expects "true"/"false"
            elif isinstance(v, float) and v.is_integer():
                v = int(v)  # write "1" not "1.0"
            f.write(f"{k}={v}\n")
    return filename
