# REHOBOAM Pairs Trading Expert Advisor

REHOBOAM is a MetaTrader 5 Expert Advisor (EA) designed for pairs trading on user-specified financial instruments. It implements a statistical arbitrage strategy based on the mean-reversion of the spread between two correlated assets. The EA calculates the hedge ratio (Beta) using historical price data, monitors the Z-Score of the spread, and opens trades when the spread deviates significantly from its mean. Position sizing is risk-based, and stop-loss/take-profit levels can be set using either Z-Score thresholds or percentage-based equity losses.

## Features

* **Pairs Trading Strategy**: Trades the spread between two assets (e.g., XAUUSD-XAGUSD, AUDUSD-NZDUSD) based on mean-reversion.
* **Hedge Ratio Calculation**: Computes Beta using linear regression over a user-defined lookback period (default: 252 days if using daily timeframe, else, 252 bars).
* **Z-Score Based Entries**: Opens long/short spread positions when the Z-Score exceeds a threshold (default: |Z| ≥ 2.0).
* **Flexible Stop-Loss Options**:

  * Z-Score based stop-loss (default: |Z| ≥ 4.0).
  * Percentage-based stop-loss (default: 2% of entry equity).

* **Risk-Based Position Sizing**: Risks a user-defined percentage of account balance (default: 1%) or uses a fixed spread move for percentage-based SL.
* **Correlation Check**: Ensures the pair has sufficient correlation (default: ≥ 0.8) as a proxy for cointegration, with an option to bypass.
* **Take-Profit**: Configurable risk-reward ratio (default: 2.0).
* **Daily Timeframe**: Optimized for D1 charts, but supports other timeframes.
* **Error Handling**: Robust checks for data availability, symbol validity, and margin requirements.



## Requirements

* **Platform**: MetaTrader 5 (MQL5)
* **Broker**: Must support hedging and provide both assets in Market Watch.
* **Account**: Minimum balance depends on pair volatility and leverage (e.g., $10,000+ for forex pairs with 1:100 leverage).
* **Libraries**: Uses MQL5's and .



## Installation

1. **Download**: Clone or download the repository from GitHub.
2. **Compile**: Copy REHOBOAM.mq5 to the Experts folder in your MetaTrader 5 data directory (MQL5/Experts/).
3. **Apply to Chart**:

   * Open MetaTrader 5.
   * Drag REHOBOAM onto a chart (any symbol/timeframe, though D1 is recommended).
   * Configure input parameters (see below).

4. **Enable Auto-Trading**: Ensure the "AutoTrading" button is active in MetaTrader 5.



## Input Parameters

ParameterTypeDefaultDescriptionSymbolAstring""First asset (e.g., XAUUSD, AUDUSD). Must be specified.SymbolBstring""Second asset (e.g., XAGUSD, NZDUSD). Must be specified.TimeframeENUM\_TIMEFRAMESPERIOD\_D1Chart timeframe (Daily recommended).LookbackPeriodint20Bars for calculating spread mean and std dev.RegressionPeriodint252Bars for hedge ratio (Beta) calculation.EntryZScoredouble1.8Z-Score threshold for trade entry (StopZScoredouble4.0Z-Score threshold for stop-loss (if SL\_Type = SL\_ZScore).RiskPercentdouble1.0% of account balance to risk per trade (if SL\_Type = SL\_ZScore).MinCorrelationdouble0.5Minimum correlation for pair to trade.BypassCorrelationCheckboolfalseIf true, skips correlation check (use cautiously).MagicNumberlong12345Unique identifier for EA's positions.RiskRewardRatiodouble2.0Take-profit as multiple of stop-loss.SL\_TypeStopLossTypeSL\_ZScoreStop-loss type: 0 (Z-Score) or 1 (Percentage).StopLossPercentdouble0.8% of entry equity for stop-loss (if SL\_Type = SL\_Percent).

## Strategy Logic

1. **Initialization**:

   * Validates symbols and data availability.
   * Calculates Beta (hedge ratio) as Cov(A,B)/Var(B) over RegressionPeriod.
   * Checks pair correlation against MinCorrelation.

2. **On Each New Bar**:

   * Computes spread (PriceA - Beta\*PriceB) using LookbackPeriod bars.
   * Calculates Z-Score: (spread - mean(spread)) / std\_dev(spread).
   * If no position:

     * If Z-Score ≤ -EntryZScore: Long spread (Buy A, Sell B).
     * If Z-Score ≥ EntryZScore: Short spread (Sell A, Buy B).

   * If position exists:

     * For SL\_ZScore: Closes if Z-Score hits stop-loss or take-profit levels.
     * For SL\_Percent: Closes if profit/loss reaches ±StopLossPercent or ±StopLossPercent\*RiskRewardRatio of entry equity.

3. **Position Sizing**:

   * For SL\_ZScore: Risks RiskPercent of balance, assuming adverse move = 2\*sigma.
   * For SL\_Percent: Risks StopLossPercent of balance.
   * Adjusts lots to broker’s min/max/step requirements.



## Usage Notes

* **Pair Selection**: Choose highly correlated pairs (e.g., XAUUSD-XAGUSD, AUDUSD-NZDUSD). Forex pairs like GBPUSD-EURUSD may have low spread volatility.
* **Backtesting**: Use “Every tick” mode, enable commissions/slippage, and test over 1-2 years. Validate on out-of-sample data to avoid over-optimization.
* **Optimization**: Focus on EntryZScore (1.8-2.2), StopLossPercent (0.25-0.4), and RiskRewardRatio (1.5-2.5). Prioritize high Sharpe Ratio (>2), low Equity DD % (<2%), and Recovery Factor (>3).
* **Risk Management**: Ensure account balance supports margin for volatile pairs. Cap lot sizes if needed (modify CalculateLots).
* **Broker Compatibility**: Verify symbols exist in Market Watch and support hedging. Check lot step sizes (e.g., 0.01 for forex).



## Example Settings

Based on optimization results (see analysis below), a robust starting point:

* SymbolA = GBPUSD, SymbolB = EURUSD
* EntryZScore = 1.8
* StopLossPercent = 0.8
* SL\_Type = SL\_Percent
* RiskRewardRatio = 2.0
* MinCorrelation = 0.5
* LookbackPeriod = 20



## Optimization Insights

From optimization data:

* **Best Settings**: EntryZScore=1.8, StopLossPercent=0.8 (Sharpe 1.45, Profit $5375, DD 9.57%, 260 trades, TEST PERIOD: Jan 1st 2024 to Aug 31st 2025).
* **Observations**:

  * Lower StopLossPercent (0.3-0.5) yields higher Sharpe/Recovery, lower DD.
  * EntryZScore 1.6-2.0 balances trade frequency and quality.
  * Higher StopLossPercent (>0.7) or EntryZScore (>2.6) often lead to losses/high DD.

* **Recommendations**:

  * Test alternative pairs with higher spread volatility (e.g., XAUUSD-XAGUSD).
  * Cap max lots in CalculateLots to avoid margin issues.



## Cointegration Screening Tool

The cointegration\_screen.py script is a Python tool designed to identify cointegrated pairs for the REHOBOAM EA, ensuring robust pair selection based on statistical tests. It processes historical price data from price\_history.csv (generated by create\_price\_history.py) and outputs results to cointegration\_results.txt and a dendrogram plot for clustering analysis. The tool is located in folder 'rehoboam-tools'.

### Overview

The tool screens for cointegrated pairs (e.g., XAUUSD-XAGUSD, AUDUSD-NZDUSD) that exhibit mean-reverting spreads, ideal for the EA’s strategy. It uses:

* **Engle-Granger Test**: Tests if the spread (PriceA - Beta\*PriceB) is stationary (p-value < 0.05).
* **Johansen Test**: Assesses multi-asset cointegration (optional, may be skipped if numerically unstable).
* **Half-Life**: Measures the speed of spread reversion (in hours for PERIOD\_H1).
* **Spread Volatility**: Quantifies spread variability for risk assessment.
* **Hierarchical Clustering**: Visualizes asset similarity via a dendrogram.



### Workflow

1. **Input**: Reads price\_history.csv (format: Date in MM/DD/YYYY HH:MM, columns for symbols like XAUUSD, XAGUSD).

   * Generated by create\_price\_history.py from MT5-exported CSVs (e.g., XAUUSD.csv).
   * Cleans data (removes NaNs/zeros, normalizes prices).

2. **Processing**:

   * Computes correlation for each pair (filters pairs with correlation < 0.8).
   * Runs Engle-Granger test to calculate Beta (hedge ratio) and p-value.
   * Calculates half-life of the spread’s mean reversion.
   * Computes spread volatility (standard deviation).
   * Performs Johansen test for multi-asset cointegration (if >2 symbols, may skip if unstable).
   * Generates a dendrogram to cluster assets by return correlation.

3. **Output**:

   * Saves results to cointegration\_results.txt with columns: Pair, Correlation, Beta, EG\_PValue, HalfLife, SpreadVolatility, optionally Johansen\_Trace, Johansen\_MaxEig.
   * Displays a dendrogram plot showing asset clusters.



### Dendrogram Interpretation

The dendrogram visualizes hierarchical clustering of assets based on their percentage returns’ correlation:

* **X-Axis**: Lists symbols (e.g., XAUUSD, XAGUSD, AUDUSD, NZDUSD).
* **Y-Axis**: Euclidean distance (lower = higher correlation).
* **Branches**: Assets merging at lower heights (e.g., Y=0.2–0.3) are highly correlated (~0.8–0.9), making them strong candidates for pairs trading.
* **Example**: If XAUUSD and XAGUSD merge at Y=0.2, and AUDUSD-NZDUSD at Y=0.3, prioritize these pairs for cointegration testing.
* **Usage**: Select pairs from the same cluster (low merge height) and cross-reference with cointegration\_results.txt for cointegration metrics.



### Optimal Parameters in cointegration\_results.txt

To select pairs for the EA, review cointegration\_results.txt for the following criteria:

* **EG\_PValue < 0.05**: Indicates the spread is stationary (cointegrated), essential for mean-reversion.
* **Correlation > 0.85**: Matches the EA’s MinCorrelation for robust pair relationships.
* **HalfLife < 100 hours**: For PERIOD\_H1, prefer pairs reverting within 50–100 hours to align with LookbackPeriod = 100–200.
* **SpreadVolatility**:

  * Commodities (e.g., XAUUSD-XAGUSD): ~$1–3 (unnormalized prices) for manageable risk.
  * Forex (e.g., AUDUSD-NZDUSD): ~0.0005–0.0015 for stable spreads.

* **Beta**: Use as the hedge ratio in the EA. Ensure it’s stable (e.g., 50–70 for XAUUSD-XAGUSD, ~1 for AUDUSD-NZDUSD).
* **Dendrogram Confirmation**: Choose pairs from closely clustered assets (low merge height) to ensure high correlation.



### Usage

1. **Generate price\_history.csv**:

   * Run create\_price\_history.py to merge MT5-exported CSVs (e.g., XAUUSD.csv, XAGUSD.csv) into price\_history.csv. Ensure you use mt5 export bars method and rename the file to the symbol you are exporting. then run the create\_price\_history.py file
   * Ensure 1–3 years of hourly data (~12,000–36,000 bars).

2. run rehoboam-tools\\cointegration\_screen.py
3. **Review Outputs**:

   * Check cointegration\_results.txt for pair metrics.
   * Analyze the dendrogram to confirm clustered pairs.

4. **Apply to EA**:

   * Set SymbolA, SymbolB to a top pair (e.g., XAUUSD-XAGUSD with EG\_PValue < 0.05, HalfLife < 100 hours).
   * Use EntryZScore = 1.8–2.0, StopLossPercent = 0.25–0.4, LookbackPeriod = 100–200.



### Notes

* **Normalization**: The tool normalizes prices to reduce scaling issues. If unnormalized results are preferred, comment out df = df / df.mean() in cointegration\_screen.py.
* **Johansen Test**: May be omitted if numerically unstable (see cointegration\_results.txt note).
* **Re-screening**: Run monthly to adapt to changing market conditions.



## Limitations

* **Correlation Proxy**: Uses correlation as a proxy for cointegration, which may miss non-stationary pairs. Consider adding ADF test for robustness.
* **Volatility Sensitivity**: Low-volatility pairs (e.g., GBPUSD-EURUSD) may lead to oversized positions.
* **Market Conditions**: Mean-reversion assumes stable pair relationships, which may break during extreme market events.
* **Broker Dependencies**: Lot sizing and margin requirements vary by broker.
* **version 1.1:** This is still a work in progress. use version 1.0



## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (git checkout -b feature/YourFeature).
3. Commit changes (git commit -m "Add YourFeature").
4. Push to the branch (git push origin feature/YourFeature).
5. Open a pull request.



Suggestions:

* Add cointegration tests (e.g., ADF).
* Implement dynamic Beta recalculation.
* Add max lot size cap in CalculateLots.



## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Disclaimer

Trading involves significant risk. Use REHOBOAM at your own risk. Backtest thoroughly and validate on a demo account before live trading. The authors and xAI are not responsible for any financial losses.

## Contact

For issues or suggestions, open an issue on GitHub or contact the maintainers.

*Copyright 2025, Stephen Njai*

