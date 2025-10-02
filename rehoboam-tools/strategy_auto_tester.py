import os
import subprocess
import datetime
import xml.etree.ElementTree as ET
import pandas as pd

# --- CONFIG ---
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
EA_NAME = "REHOBOAM_v1.0.ex5"
SYMBOL = "GBPUSD"
TIMEFRAME = "M1"
DEPOSIT = 15000
LEVERAGE = 100
SERVER = "MetaQuotes-Demo"
LOGIN = "5040745538"
PASSWORD = "LfJdNv_3"
RESULTS_DIR = "optimization_results"

# --- Optimization date range (easy to edit) ---
START_DATE = datetime.date(2025, 9, 1)
END_DATE   = datetime.date(2025, 9, 30)
WEEKLY_STEP = datetime.timedelta(days=1)

# --- Ensure results folder exists ---
os.makedirs(RESULTS_DIR, exist_ok=True)


def write_ini(from_date, to_date, ini_file, result_file):
    """Write MT5 config ini for optimization (MT5-compliant format).
    if you want to optimize a parameter, chnage the N to a Y and change the {default, start, stop, step, N/Y} to your liking"""
    config = f"""
[Common]
Login={LOGIN}
Password={PASSWORD}
Server={SERVER}

[Tester]
Expert={EA_NAME}
Symbol={SYMBOL}
Period={TIMEFRAME}
Model=4
ExecutionMode=221
Optimization=1
ForwardMode=0
Deposit={DEPOSIT}
Currency=USD
Leverage={LEVERAGE}
FromDate={from_date.strftime("%Y.%m.%d")}
ToDate={to_date.strftime("%Y.%m.%d")}
Report={result_file}
ReplaceReport=1
ShutdownTerminal=0
OptimizationCriterion=5

[TesterInputs]
SymbolA=GBPUSD
SymbolB=EURUSD
Timeframe=1||0||0||49153||N
LookbackPeriod=20||20||1||200||N
RegressionPeriod=252||252||1||2520||N
EntryZScore=6.6||0||0.1||10||Y
StopZScore=5||0||0.1||10||Y
RiskPercent=1.0||1.0||0.100000||10.000000||N
MinCorrelation=0.2||0.2||0.020000||2.000000||N
BypassCorrelationCheck=true||false||0||true||N
MagicNumber=12345||12345||1||123450||N
RiskRewardRatio=2.0||2.0||0.200000||20.000000||N
SL_Type=0||0||0||1||N
StopLossPercent=1.2||0||0.2||5||N
MaxLots=5.0||5.0||0.500000||50.000000||N
"""
    with open(ini_file, "w", encoding="utf-8") as f:
        f.write(config.strip())


def run_mt5(ini_file):
    """Launch MT5 with ini file."""
    datadir = r"MT5_data"
    os.makedirs(datadir, exist_ok=True)
    cmd = [MT5_PATH, f"/config:{ini_file}", f"/datadir:{datadir}"]
    subprocess.run(cmd)


def xml_to_csv(xml_file, csv_file):
    """Convert MT5 optimization XML -> CSV."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    rows = []
    for strat in root.findall("Strategy"):
        row = {k: v for k, v in strat.attrib.items()}
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"✅ Saved: {csv_file}")
    else:
        print(f"⚠ No strategies found in {xml_file}")


# --- MAIN LOOP ---
week = 1
curr_date = START_DATE

while curr_date < END_DATE:
    week_start = curr_date
    week_end = min(curr_date + WEEKLY_STEP, END_DATE)

    ini_file = f"opt_config_week{week}.ini"
    result_file = os.path.join(RESULTS_DIR, f"weekly_result_week{week}.xml")
    csv_file = os.path.join(RESULTS_DIR, f"week_{week}_{week_start}_{week_end}.csv")

    print(f"\n🚀 Running optimization for Week {week}: {week_start} -> {week_end}")

    # Write config file
    write_ini(week_start, week_end, ini_file, result_file)

    # Run MT5
    run_mt5(ini_file)

    # Convert result XML -> CSV
    if os.path.exists(result_file):
        xml_to_csv(result_file, csv_file)
    else:
        print(f"⚠ No result file found for Week {week}")

    # Cleanup .ini file after run
    if os.path.exists(ini_file):
        os.remove(ini_file)
        print(f"🧹 Cleaned up: {ini_file}")

    curr_date += WEEKLY_STEP
    week += 1
