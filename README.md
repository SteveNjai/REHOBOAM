REHOBOAM Pairs Trading Expert Advisor
=====================================

REHOBOAM is a MetaTrader 5 Expert Advisor (EA) designed for pairs trading on user-specified financial instruments. Version 4.0 introduces significant enhancements over previous versions, including Monte Carlo simulation for dynamic Z-score optimization, flexible simulation intervals, selective logging for performance, and the ability to use a static Z-score. The EA implements a statistical arbitrage strategy based on the mean-reversion of the spread between two correlated assets, calculating the hedge ratio (Beta) using resampled tick data and executing trades based on Z-score thresholds. Position sizing is risk-based, with configurable stop-loss and take-profit options.

Features
--------

*   **Pairs Trading Strategy**: Trades the spread between two assets (e.g., GBPUSD-EURUSD, XAUUSD-XAGUSD) based on mean-reversion.
    
*   **Hedge Ratio Calculation**: Computes Beta using linear regression over a user-defined lookback period (default: 7020 resampled bars, ~2s intervals).
    
*   **Dynamic Z-Score Optimization**: Uses Monte Carlo simulation to optimize the entry Z-score, updated periodically (default: every 240 minutes), with an option to disable via DynamicEntryZScore=false for a static Z-score (default: 9.0).
    
*   **Flexible Stop-Loss Options**:
    
    *   Z-Score based stop-loss (default: |Z| ≥ 4.8).
        
    *   Percentage-based stop-loss (default: 2.5% of entry equity).
        
*   **Risk-Based Position Sizing**: Risks a user-defined percentage of account balance (default: 1%) based on 2\*sigma for Z-score stops or a fixed percentage for equity-based stops.
    
*   **Correlation Check**: Ensures sufficient pair correlation (default: ≥ 0.2) as a proxy for cointegration, with an option to bypass.
    
*   **Take-Profit**: Configurable as a risk-reward ratio (default: 2.0) or Z-score threshold.
    
*   **Resampled Tick Data**: Uses high-resolution tick data (default: 2s bars) for spread calculations, falling back to M1 if insufficient ticks.
    
*   **Selective Logging**: Logs only critical events (initialization, errors, simulation results, trades) to keep log files small ({SymbolA}-{SymbolB}-rehoboam-v4.log), with an option to disable file logging (EnableLogging=false) for faster backtests.
    
*   **Simulation Control**: Configurable simulation interval (SimIntervalMinutes) and the ability to disable dynamic Z-score updates for performance.
    
*   **Error Handling**: Robust checks for data availability, symbol validity, and margin requirements.
    
*   **Output Files**: Saves simulation results to {SymbolA}-{SymbolB}-rehoboam-v4-results.txt in the MQL5/Files directory.
    

Requirements
------------

*   **Platform**: MetaTrader 5 (MQL5)
    
*   **Broker**: Must support hedging and provide both assets in Market Watch.
    
*   **Account**: Minimum balance depends on pair volatility and leverage (e.g., $10,000+ for forex pairs with 1:100 leverage).
    
*   **Libraries**: Uses MQL5's , , and .
    

Installation
------------

1.  **Download**: Clone or download the repository from GitHub.
    
2.  **Compile**: Copy REHOBOAM-v4.mq5 to the Experts folder in your MetaTrader 5 data directory (MQL5/Experts/).
    
3.  **Apply to Chart**:
    
    *   Open MetaTrader 5.
        
    *   Drag REHOBOAM onto a chart (any symbol/timeframe, though M1 is recommended for tick resampling).
        
    *   Configure input parameters (see below).
        
4.  **Enable Auto-Trading**: Ensure the "AutoTrading" button is active in MetaTrader 5.
    

Input Parameters
----------------

ParameterTypeDefaultDescriptionSymbolAstring"GBPUSD"First asset (e.g., GBPUSD, XAUUSD). Must be specified.SymbolBstring"EURUSD"Second asset (e.g., EURUSD, XAGUSD). Must be specified.RESAMPLINGint2Resampling period for tick data (seconds).LookbackPeriodint5720Bars for calculating spread mean and std dev (resampled bars).RegressionPeriodint7020Bars for hedge ratio (Beta) calculation (resampled bars).DefaultEntryZScoredouble9.0Default Z-Score threshold for trade entry if DynamicEntryZScore=false.StopZScoredouble4.8Z-Score threshold for stop-loss (if SL\_Type = SL\_ZScore).TakeProfitZScoredouble0.0Z-Score threshold for take-profit (if TP\_Type = TP\_ZScore).RiskPercentdouble1.0% of account balance to risk per trade (if SL\_Type = SL\_ZScore).MinCorrelationdouble0.2Minimum correlation for pair to trade.BypassCorrelationCheckboolfalseIf true, skips correlation check (use cautiously).BypassCointCheckbooltrueIf true, skips cointegration check (default: true for flexibility).MagicNumberlong1099Unique identifier for EA's positions.RiskRewardRatiodouble2.0Take-profit as multiple of stop-loss (if TP\_Type = TP\_Multiple).SL\_TypeStopLossTypeSL\_ZScoreStop-loss type: 0 (Z-Score) or 1 (Percentage).TP\_TypeTakeProfitTypeTP\_MultipleTake-profit type: 0 (Multiple) or 1 (Z-Score).StopLossPercentdouble2.5% of entry equity for stop-loss (if SL\_Type = SL\_Percent).MaxLotsdouble1.0Maximum lot size per trade.SimNPathsint50Number of Monte Carlo simulation paths.SimNStepsint1800Number of steps per simulation path (~1hr at 2s resampling).ZScoreMaxdouble5.0Maximum Z-score for simulation testing.ZScoreStepdouble0.2Z-score increment for simulation testing.SimIntervalMinutesdouble240.0Interval for re-running Z-score simulation (minutes).LotSizedouble0.1Lot size for simulation calculations.PipValuedouble10.0USD per pip for 1 lot in simulations.PipSizedouble0.0001Pip size for calculations (e.g., 0.0001 for forex).EnableLoggingbooltrueEnable logging to file ({SymbolA}-{SymbolB}-rehoboam-v4.log).DynamicEntryZScorebooltrueIf true, uses Monte Carlo simulation to optimize Z-score; if false, uses DefaultEntryZScore.

Strategy Logic
--------------

1.  **Initialization**:
    
    *   Validates symbols and data availability.
        
    *   Calculates Beta (hedge ratio) as Cov(A,B)/Var(B) over RegressionPeriod resampled bars.
        
    *   Checks pair correlation against MinCorrelation (unless bypassed).
        
    *   If DynamicEntryZScore=true, runs Monte Carlo simulation to optimize EntryZScore and sets a timer for periodic updates (SimIntervalMinutes). If false, sets EntryZScore = DefaultEntryZScore.
        
2.  **On Each Tick** (resampled every RESAMPLING seconds):
    
    *   Computes spread (PriceA - Beta\*PriceB) using LookbackPeriod resampled bars.
        
    *   Calculates Z-Score: (spread - mean(spread)) / std\_dev(spread).
        
    *   If no position:
        
        *   If Z-Score ≤ -EntryZScore: Long spread (Buy A, Sell B).
            
        *   If Z-Score ≥ EntryZScore: Short spread (Sell A, Buy B).
            
    *   If position exists:
        
        *   For SL\_ZScore: Closes if Z-Score hits stop-loss (StopZScore) or take-profit (TakeProfitZScore or RiskRewardRatio-based).
            
        *   For SL\_Percent: Closes if profit/loss reaches ±StopLossPercent or ±StopLossPercent\*RiskRewardRatio of entry equity.
            
3.  **Position Sizing**:
    
    *   For SL\_ZScore: Risks RiskPercent of balance, assuming adverse move = 2\*sigma.
        
    *   For SL\_Percent: Risks StopLossPercent of balance.
        
    *   Caps lot sizes at MaxLots and adjusts to broker’s min/max/step requirements.
        
4.  **Simulation (if DynamicEntryZScore=true)**:
    
    *   Runs Monte Carlo simulation with SimNPaths paths and SimNSteps steps to test Z-scores from 0 to ZScoreMax in ZScoreStep increments.
        
    *   Selects the Z-score with the highest Sharpe ratio (or highest mean PNL if Sharpe is undefined), capped at 8.0.
        
    *   Writes results to {SymbolA}-{SymbolB}-rehoboam-v4-results.txt.
        
5.  **Logging**:
    
    *   Logs critical events (initialization, errors, simulation results, trades) to {SymbolA}-{SymbolB}-rehoboam-v4.log when EnableLogging=true.
        
    *   Always prints all events to the MT5 Experts log for debugging.
        

Usage Notes
-----------

*   **Pair Selection**: Choose highly correlated pairs (e.g., XAUUSD-XAGUSD, AUDUSD-NZDUSD, GBPUSD-EURUSD). Use the cointegration\_screen.py tool to identify cointegrated pairs.
    
*   **Backtesting**: Use “Every tick” mode, enable commissions/slippage, and test over 1-2 years. Set DynamicEntryZScore=false for faster backtests or true for dynamic optimization.
    
*   **Optimization**: Focus on DefaultEntryZScore (8.0-10.0), StopLossPercent (0.5-2.5), RiskRewardRatio (1.5-2.5), and SimIntervalMinutes (5.0-240.0). Prioritize high Sharpe Ratio (>2), low Equity DD % (<2%), and Recovery Factor (>3).
    
*   **Risk Management**: Ensure sufficient account balance for volatile pairs (e.g., XAUUSD-XAGUSD). Adjust MaxLots to avoid margin issues.
    
*   **Broker Compatibility**: Verify symbols exist in Market Watch, support hedging, and have appropriate lot step sizes (e.g., 0.01 for forex).
    
*   **Performance**: Disable logging (EnableLogging=false) and simulations (DynamicEntryZScore=false) for faster backtests. Use smaller SimIntervalMinutes (e.g., 5.0) for frequent updates on powerful systems.
    

Example Settings
----------------

Based on optimization results for version 4.0:

*   **SymbolA** = GBPUSD, **SymbolB** = EURUSD
    
*   **RESAMPLING** = 2 (seconds)
    
*   **DefaultEntryZScore** = 9.0
    
*   **DynamicEntryZScore** = true
    
*   **SimIntervalMinutes** = 240.0
    
*   **StopLossPercent** = 0.8
    
*   **SL\_Type** = SL\_Percent
    
*   **RiskRewardRatio** = 2.0
    
*   **MinCorrelation** = 0.2
    
*   **LookbackPeriod** = 5720
    
*   **RegressionPeriod** = 7020
    
*   **EnableLogging** = true
    

Optimization Insights
---------------------

From optimization data (Jan 1, 2024, to Aug 31, 2025):

*   **Best Settings**:
    
    *   DynamicEntryZScore=true, SimIntervalMinutes=240.0, StopLossPercent=0.8, DefaultEntryZScore=9.0 (Sharpe 1.45, Profit $5375, DD 9.57%, 260 trades).
        
    *   Dynamic Z-score optimization typically yields EntryZScore between 2.0-8.0, improving trade quality.
        
*   **Observations**:
    
    *   DynamicEntryZScore=true enhances adaptability in volatile markets but increases computation time.
        
    *   DynamicEntryZScore=false with DefaultEntryZScore=9.0 is faster and suitable for stable pairs like GBPUSD-EURUSD.
        
    *   Lower StopLossPercent (0.3-0.8) reduces drawdowns and improves Sharpe/Recovery.
        
    *   SimIntervalMinutes=5.0-60.0 balances responsiveness and performance on powerful systems.
        
*   **Recommendations**:
    
    *   Test high-volatility pairs (e.g., XAUUSD-XAGUSD) with DynamicEntryZScore=true.
        
    *   Use DynamicEntryZScore=false for quick backtests or low-volatility pairs.
        
    *   Cap MaxLots to manage margin for volatile pairs.
        

Cointegration Screening Tool
----------------------------

The cointegration\_screen.py script is a Python tool designed to identify cointegrated pairs for REHOBOAM, ensuring robust pair selection based on statistical tests. It processes historical price data from price\_history.csv (generated by create\_price\_history.py) and outputs results to cointegration\_results.txt and a dendrogram plot for clustering analysis. The tool, located in the rehoboam-tools folder, is fully compatible with version 4.0.

### Overview

The tool screens for cointegrated pairs (e.g., XAUUSD-XAGUSD, AUDUSD-NZDUSD) that exhibit mean-reverting spreads, ideal for the EA’s strategy. It uses:

*   **Engle-Granger Test**: Tests if the spread (PriceA - Beta\*PriceB) is stationary (p-value < 0.05).
    
*   **Johansen Test**: Assesses multi-asset cointegration (optional, may be skipped if numerically unstable).
    
*   **Half-Life**: Measures the speed of spread reversion (in hours for PERIOD\_H1).
    
*   **Spread Volatility**: Quantifies spread variability for risk assessment.
    
*   **Hierarchical Clustering**: Visualizes asset similarity via a dendrogram.
    

### Workflow

1.  **Input**: Reads price\_history.csv (format: Date in MM/DD/YYYY HH:MM, columns for symbols like XAUUSD, XAGUSD).
    
    *   Generated by create\_price\_history.py from MT5-exported CSVs (e.g., XAUUSD.csv).
        
    *   Cleans data (removes NaNs/zeros, normalizes prices).
        
2.  **Processing**:
    
    *   Computes correlation for each pair (filters pairs with correlation < 0.8).
        
    *   Runs Engle-Granger test to calculate Beta and p-value.
        
    *   Calculates half-life of the spread’s mean reversion.
        
    *   Computes spread volatility (standard deviation).
        
    *   Performs Johansen test for multi-asset cointegration (if >2 symbols, may skip if unstable).
        
    *   Generates a dendrogram to cluster assets by return correlation.
        
3.  **Output**:
    
    *   Saves results to cointegration\_results.txt with columns: Pair, Correlation, Beta, EG\_PValue, HalfLife, SpreadVolatility, optionally Johansen\_Trace, Johansen\_MaxEig.
        
    *   Displays a dendrogram plot showing asset clusters.
        

### Dendrogram Interpretation

The dendrogram visualizes hierarchical clustering of assets based on their percentage returns’ correlation:

*   **X-Axis**: Lists symbols (e.g., XAUUSD, XAGUSD, AUDUSD, NZDUSD).
    
*   **Y-Axis**: Euclidean distance (lower = higher correlation).
    
*   **Branches**: Assets merging at lower heights (e.g., Y=0.2–0.3) are highly correlated (~0.8–0.9), making them strong candidates for pairs trading.
    
*   **Example**: If XAUUSD and XAGUSD merge at Y=0.2, and AUDUSD-NZDUSD at Y=0.3, prioritize these pairs for cointegration testing.
    
*   **Usage**: Select pairs from the same cluster (low merge height) and cross-reference with cointegration\_results.txt for cointegration metrics.
    

### Optimal Parameters in cointegration\_results.txt

To select pairs for the EA, review cointegration\_results.txt for the following criteria:

*   **EG\_PValue < 0.05**: Indicates the spread is stationary (cointegrated), essential for mean-reversion.
    
*   **Correlation > 0.85**: Matches the EA’s MinCorrelation for robust pair relationships (though version 4.0 allows lower values, e.g., 0.2).
    
*   **HalfLife < 100 hours**: For PERIOD\_H1, prefer pairs reverting within 50–100 hours to align with LookbackPeriod = 5720 (resampled bars).
    
*   **SpreadVolatility**:
    
    *   Commodities (e.g., XAUUSD-XAGUSD): ~$1–3 (unnormalized prices) for manageable risk.
        
    *   Forex (e.g., AUDUSD-NZDUSD): ~0.0005–0.0015 for stable spreads.
        
*   **Beta**: Use as the hedge ratio in the EA. Ensure it’s stable (e.g., 50–70 for XAUUSD-XAGUSD, ~1 for AUDUSD-NZDUSD).
    
*   **Dendrogram Confirmation**: Choose pairs from closely clustered assets (low merge height) to ensure high correlation.
    

### Usage

1.  **Generate price\_history.csv**:
    
    *   Run create\_price\_history.py to merge MT5-exported CSVs (e.g., XAUUSD.csv, XAGUSD.csv) into price\_history.csv. Ensure you use MT5’s export bars method and rename files to match symbol names.
        
    *   Ensure 1–3 years of hourly data (~12,000–36,000 bars).
        
2.  Run rehoboam-tools\\cointegration\_screen.py.
    
3.  **Review Outputs**:
    
    *   Check cointegration\_results.txt for pair metrics.
        
    *   Analyze the dendrogram to confirm clustered pairs.
        
4.  **Apply to EA**:
    
    *   Set SymbolA, SymbolB to a top pair (e.g., XAUUSD-XAGUSD with EG\_PValue < 0.05, HalfLife < 100 hours).
        
    *   Use DynamicEntryZScore=true for adaptive trading or false for static Z-score.
        
    *   Set LookbackPeriod=5720, RegressionPeriod=7020, SimIntervalMinutes=5.0-240.0.
        

### Notes

*   **Normalization**: The tool normalizes prices to reduce scaling issues. Comment out df = df / df.mean() in cointegration\_screen.py for unnormalized results.
    
*   **Johansen Test**: May be omitted if numerically unstable (see cointegration\_results.txt note).
    
*   **Re-screening**: Run monthly to adapt to changing market conditions.
    

Limitations
-----------

*   **Correlation Proxy**: Version 4.0 allows lower MinCorrelation (0.2) and bypasses cointegration checks by default (BypassCointCheck=true), which may risk non-stationary pairs. Use the cointegration screening tool for robust pair selection.
    
*   **Volatility Sensitivity**: Low-volatility pairs (e.g., GBPUSD-EURUSD) may lead to oversized positions; adjust MaxLots accordingly.
    
*   **Market Conditions**: Mean-reversion assumes stable pair relationships, which may break during extreme market events.
    
*   **Broker Dependencies**: Lot sizing and margin requirements vary by broker.
    
*   **Performance**: Simulations (DynamicEntryZScore=true) may slow backtests; set DynamicEntryZScore=false and EnableLogging=false for faster testing.
    

Contributing
------------

Contributions are welcome! Please:

1.  Fork the repository.
    
2.  Create a feature branch (git checkout -b feature/YourFeature).
    
3.  Commit changes (git commit -m "Add YourFeature").
    
4.  Push to the branch (git push origin feature/YourFeature).
    
5.  Open a pull request.
    

Suggestions:

*   Add advanced cointegration tests (e.g., ADF) to the EA.
    
*   Implement dynamic Beta recalculation during trading.
    
*   Enhance simulation efficiency for larger SimNPaths or SimNSteps.
    

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

Disclaimer
----------

Trading involves significant risk. Use REHOBOAM at your own risk. Backtest thoroughly and validate on a demo account before live trading. The authors and xAI are not responsible for any financial losses.

Contact
-------

For issues or suggestions, open an issue on GitHub or contact the maintainers.

_Copyright 2025, Stephen Njai_
