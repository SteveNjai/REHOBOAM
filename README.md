# REHOBOAM Pairs Trading Expert Advisor

REHOBOAM is a MetaTrader 5 Expert Advisor (EA) designed for pairs trading on user-specified financial instruments. It implements a statistical arbitrage strategy based on the mean-reversion of the spread between two correlated assets. The EA calculates the hedge ratio (Beta) using historical price data, monitors the Z-Score of the spread, and opens trades when the spread deviates significantly from its mean. Position sizing is risk-based, and stop-loss/take-profit levels can be set using either Z-Score thresholds or percentage-based equity losses.

## Features
- **Pairs Trading Strategy**: Trades the spread between two assets (e.g., XAUUSD-XAGUSD, AUDUSD-NZDUSD) based on mean-reversion.
- **Hedge Ratio Calculation**: Computes Beta using linear regression over a user-defined lookback period (default: 252 days).
- **Z-Score Based Entries**: Opens long/short spread positions when the Z-Score exceeds a threshold (default: |Z| ≥ 2.0).
- **Flexible Stop-Loss Options**:
  - Z-Score based stop-loss (default: |Z| ≥ 4.0).
  - Percentage-based stop-loss (default: 2% of entry equity).
- **Risk-Based Position Sizing**: Risks a user-defined percentage of account balance (default: 1%) or uses a fixed spread move for percentage-based SL.
- **Correlation Check**: Ensures the pair has sufficient correlation (default: ≥ 0.8) as a proxy for cointegration, with an option to bypass.
- **Take-Profit**: Configurable risk-reward ratio (default: 2.0).
- **Daily Timeframe**: Optimized for D1 charts, but supports other timeframes.
- **Error Handling**: Robust checks for data availability, symbol validity, and margin requirements.

## Requirements
- **Platform**: MetaTrader 5 (MQL5)
- **Broker**: Must support hedging and provide both assets in Market Watch.
- **Account**: Minimum balance depends on pair volatility and leverage (e.g., $10,000+ for forex pairs with 1:100 leverage).
- **Libraries**: Uses MQL5's `<Math\Stat\Math.mqh>` and `<Trade\Trade.mqh>`.

## Installation
1. **Download**: Clone or download the repository from GitHub.
2. **Compile**: Copy `REHOBOAM.mq5` to the `Experts` folder in your MetaTrader 5 data directory (`MQL5/Experts/`).
3. **Apply to Chart**:
   - Open MetaTrader 5.
   - Drag REHOBOAM onto a chart (any symbol/timeframe, though D1 is recommended).
   - Configure input parameters (see below).
4. **Enable Auto-Trading**: Ensure the "AutoTrading" button is active in MetaTrader 5.

## Input Parameters
| Parameter               | Type        | Default | Description                                                                 |
|-------------------------|-------------|---------|-----------------------------------------------------------------------------|
| `SymbolA`               | string      | ""      | First asset (e.g., XAUUSD, AUDUSD). Must be specified.                      |
| `SymbolB`               | string      | ""      | Second asset (e.g., XAGUSD, NZDUSD). Must be specified.                     |
| `Timeframe`             | ENUM_TIMEFRAMES | PERIOD_D1 | Chart timeframe (Daily recommended).                              |
| `LookbackPeriod`        | int         | 20      | Bars for calculating spread mean and std dev.                        |
| `RegressionPeriod`      | int         | 252     | Bars for hedge ratio (Beta) calculation.                             |
| `EntryZScore`           | double      | 1.8     | Z-Score threshold for trade entry (|Z| ≥ this value).                |
| `StopZScore`            | double      | 4.0     | Z-Score threshold for stop-loss (if `SL_Type = SL_ZScore`).         |
| `RiskPercent`           | double      | 1.0     | % of account balance to risk per trade (if `SL_Type = SL_ZScore`).  |
| `MinCorrelation`        | double      | 0.5     | Minimum correlation for pair to trade.                               |
| `BypassCorrelationCheck`| bool        | false   | If true, skips correlation check (use cautiously).                   |
| `MagicNumber`           | long        | 12345   | Unique identifier for EA's positions.                                |
| `RiskRewardRatio`       | double      | 2.0     | Take-profit as multiple of stop-loss.                                |
| `SL_Type`               | StopLossType| SL_ZScore | Stop-loss type: 0 (Z-Score) or 1 (Percentage).                    |
| `StopLossPercent`       | double      | 0.8     | % of entry equity for stop-loss (if `SL_Type = SL_Percent`).        |
| `ExpectedAdverseSpread` | double      | 0.01    | Expected spread move for sizing (if `SL_Type = SL_Percent`).        |

## Strategy Logic
1. **Initialization**:
   - Validates symbols and data availability.
   - Calculates Beta (hedge ratio) as Cov(A,B)/Var(B) over `RegressionPeriod`.
   - Checks pair correlation against `MinCorrelation`.

2. **On Each New Bar**:
   - Computes spread (PriceA - Beta*PriceB) using `LookbackPeriod` bars.
   - Calculates Z-Score: `(spread - mean(spread)) / std_dev(spread)`.
   - If no position:
     - If Z-Score ≤ -`EntryZScore`: Long spread (Buy A, Sell B).
     - If Z-Score ≥ `EntryZScore`: Short spread (Sell A, Buy B).
   - If position exists:
     - For `SL_ZScore`: Closes if Z-Score hits stop-loss or take-profit levels.
     - For `SL_Percent`: Closes if profit/loss reaches ±`StopLossPercent` or ±`StopLossPercent*RiskRewardRatio` of entry equity.

3. **Position Sizing**:
   - For `SL_ZScore`: Risks `RiskPercent` of balance, assuming adverse move = 2*sigma.
   - For `SL_Percent`: Risks `StopLossPercent` of balance, assuming adverse move = `ExpectedAdverseSpread`.
   - Adjusts lots to broker’s min/max/step requirements.

## Usage Notes
- **Pair Selection**: Choose highly correlated pairs (e.g., XAUUSD-XAGUSD, AUDUSD-NZDUSD). Forex pairs like GBPUSD-EURUSD may have low spread volatility, requiring careful tuning of `ExpectedAdverseSpread`.
- **Backtesting**: Use “Every tick” mode, enable commissions/slippage, and test over 1-2 years. Validate on out-of-sample data to avoid over-optimization.
- **Optimization**: Focus on `EntryZScore` (1.8-2.2), `StopLossPercent` (0.25-0.4), `ExpectedAdverseSpread` (0.005-0.02), and `RiskRewardRatio` (1.5-2.5). Prioritize high Sharpe Ratio (>2), low Equity DD % (<2%), and Recovery Factor (>3).
- **Risk Management**: Ensure account balance supports margin for volatile pairs. Cap lot sizes if needed (modify `CalculateLots`).
- **Broker Compatibility**: Verify symbols exist in Market Watch and support hedging. Check lot step sizes (e.g., 0.01 for forex).

## Example Settings
Based on optimization results (see analysis below), a robust starting point:
- `SymbolA = XAUUSD`, `SymbolB = XAGUSD`
- `EntryZScore = 2.0`
- `StopLossPercent = 0.3`
- `SL_Type = SL_Percent`
- `ExpectedAdverseSpread = 0.01` (tune based on pair’s historical spread volatility)
- `RiskRewardRatio = 2.0`
- `MinCorrelation = 0.8`
- `LookbackPeriod = 20`

## Optimization Insights
From optimization data:
- **Best Settings**: EntryZScore=2.0, StopLossPercent=0.3 (Sharpe 2.69, Profit $534, DD 1.01%, 26 trades).
- **Observations**:
  - Lower `StopLossPercent` (0.3-0.5) yields higher Sharpe/Recovery, lower DD.
  - `EntryZScore` 1.6-2.0 balances trade frequency and quality.
  - Higher `StopLossPercent` (>0.7) or `EntryZScore` (>2.6) often lead to losses/high DD.
- **Recommendations**:
  - Optimize `ExpectedAdverseSpread` (0.005-0.02) for `SL_Percent` to control sizing.
  - Test alternative pairs with higher spread volatility (e.g., XAUUSD-XAGUSD).
  - Cap max lots in `CalculateLots` to avoid margin issues.

## Limitations
- **Correlation Proxy**: Uses correlation as a proxy for cointegration, which may miss non-stationary pairs. Consider adding ADF test for robustness.
- **Volatility Sensitivity**: Low-volatility pairs (e.g., GBPUSD-EURUSD) may lead to oversized positions. Tune `ExpectedAdverseSpread` carefully.
- **Market Conditions**: Mean-reversion assumes stable pair relationships, which may break during extreme market events.
- **Broker Dependencies**: Lot sizing and margin requirements vary by broker.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Suggestions:
- Add cointegration tests (e.g., ADF).
- Implement dynamic Beta recalculation.
- Add max lot size cap in `CalculateLots`.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Disclaimer
Trading involves significant risk. Use REHOBOAM at your own risk. Backtest thoroughly and validate on a demo account before live trading. The authors and xAI are not responsible for any financial losses.

## Contact
For issues or suggestions, open an issue on GitHub or contact the maintainers.

---
*Copyright 2025, xAI - Grok Implementation*  
*https://x.ai/*
