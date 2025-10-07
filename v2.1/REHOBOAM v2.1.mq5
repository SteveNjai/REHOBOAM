//+------------------------------------------------------------------+
//|                                             REHOBOAM-v2.1.mq5    |
//|                        Copyright 2025, Stephen Njai              |
//|                                             https://github.com/SteveNjai |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Stephen Njai"
#property link      "https://github.com/SteveNjai"
#property version   "2.1"
#property strict
#property description "Pairs Trading EA utilizing correlation between pairs for user-specified symbols."
#property description "Uses daily or 1H timeframe. Hedge ratio calculated on init with 252 days."
#property description "Position sizing risks 1% of balance based on 2*sigma for stop at +/-4 Z or percentage based stop loss."
#property description "Uses correlation as a proxy for cointegration."
#property description "v2.1: Added rolling hedge ratio updates, ADF stationarity test with dynamic lookback shortening,"
#property description "EWMA for efficient mu/sigma updates, median for central tendency, and MAD for robust dispersion."

#include <Math\Stat\Math.mqh>
#include <Trade\Trade.mqh>

// Stop Loss Types
enum StopLossType
  {
   SL_ZScore = 0,   // Z-Score based stop loss
   SL_Percent = 1   // Percentage based stop loss
  };

// Take Profit Types
enum TakeProfitType
  {
   TP_Multiple = 0, // Multiple of stop loss
   TP_ZScore = 1    // Z-Score based take profit
  };

// Central Tendency Types
enum CentralTendencyType
  {
   CT_Mean = 0,     // Arithmetic mean
   CT_Median = 1    // Median (robust to skewness)
  };

// Dispersion Types
enum DispersionType
  {
   DT_StdDev = 0,   // Standard deviation
   DT_MAD = 1       // Median Absolute Deviation (robust)
  };

// Inputs
input string SymbolA = "";           // Symbol for Asset A (e.g., XAUUSD, leave empty to input manually)
input string SymbolB = "";           // Symbol for Asset B (e.g., XAGUSD, leave empty to input manually)
input ENUM_TIMEFRAMES Timeframe = PERIOD_H1; // Timeframe (Daily recommended)
input int LookbackPeriod = 20;       // Lookback for mean and std dev of spread
input int RegressionPeriod = 252;    // Lookback for hedge ratio calculation
input double EntryZScore = 2.0;      // Entry threshold for |Z-Score|
input double StopZScore = 4.0;       // Stop-loss threshold for |Z-Score| (used if SL_ZScore)
input double TakeProfitZScore = 0.0; // Take profit Z-score threshold (used if TP_ZScore, e.g., 0.0 for mean)
input double RiskPercent = 1.0;      // Risk % of account balance per trade (for sizing)
input double MinCorrelation = 0.2;   // Minimum correlation to allow trading
input bool BypassCorrelationCheck = false; // Bypass correlation check for testing (use with caution)
input long MagicNumber = 12345;      // Magic number for positions
input double RiskRewardRatio = 2.0;  // Take profit multiple of stop loss (used if TP_Multiple)
input StopLossType SL_Type = SL_Percent; // Stop loss type
input TakeProfitType TP_Type = TP_Multiple; // Take profit type
input double StopLossPercent = 2.5;  // Stop loss % of entry equity (used if SL_Percent)
input double MaxLots = 5.0;          // Maximum lot size per trade to prevent margin issues

// New Enhancements Inputs
input bool UseRollingHedge = true;   // Enable rolling hedge ratio updates
input int HedgeUpdateBars = 1;       // Recalculate hedge every N bars (1 = every bar)
input bool UseADFTest = true;        // Enable ADF stationarity test on spreads
input double ADF_PThreshold = 0.05;  // ADF p-value threshold (non-stationary if > this)
input int MinLookback = 10;          // Minimum lookback if ADF shortens it
input bool UseEWMA = true;           // Use EWMA for mu/sigma updates (efficient)
input double Alpha = 0.05;           // EWMA alpha (decay factor; smaller = smoother)
input CentralTendencyType CT_Type = CT_Median; // Central tendency (median for robustness)
input DispersionType DT_Type = DT_MAD; // Dispersion measure (MAD for robustness)

// Globals
double Beta = 0.0;
CTrade Trade;
double EntryZ = 0.0;                 // Entry Z-Score
double EntryEquity = 0.0;            // Equity at entry

// EWMA Globals
double EW_Mu = 0.0;
double EW_Sigma = 0.0;
double EW_Var = 0.0;
bool EW_Initialized = false;

// Rolling Hedge Globals
datetime LastHedgeTime = 0;
int DynamicLookback = 0;             // Adjusted lookback based on ADF

//+------------------------------------------------------------------+
//| Check if market is open for both symbols                         |
//+------------------------------------------------------------------+
bool IsMarketOpen()
  {
   bool marketOpenA = SymbolInfoInteger(SymbolA, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED;
   bool marketOpenB = SymbolInfoInteger(SymbolB, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED;
   
   if(!marketOpenA || !marketOpenB)
     {
      if(!marketOpenA) Print("Market closed for ", SymbolA);
      if(!marketOpenB) Print("Market closed for ", SymbolB);
      return false;
     }
   
   return true;
  }

//+------------------------------------------------------------------+
//| Custom covariance function                                       |
//+------------------------------------------------------------------+
double MathCovariance(double &array1[], double &array2[])
  {
   if(ArraySize(array1) != ArraySize(array2) || ArraySize(array1) == 0)
     {
      Print("Error: Arrays have different sizes or are empty.");
      return 0.0;
     }
   
   int n = ArraySize(array1);
   double mean1 = MathMean(array1);
   double mean2 = MathMean(array2);
   double sum = 0.0;
   
   for(int i = 0; i < n; i++)
     {
      sum += (array1[i] - mean1) * (array2[i] - mean2);
     }
   
   return sum / (n - 1);
  }

//+------------------------------------------------------------------+
//| Custom correlation function                                      |
//+------------------------------------------------------------------+
double MathCorrelation(double &array1[], double &array2[])
  {
   if(ArraySize(array1) != ArraySize(array2) || ArraySize(array1) == 0)
     {
      Print("Error: Arrays have different sizes or are empty in MathCorrelation.");
      return 0.0;
     }
   
   int n = ArraySize(array1);
   double cov = MathCovariance(array1, array2);
   double std1 = MathStandardDeviation(array1);
   double std2 = MathStandardDeviation(array2);
   
   if(std1 == 0 || std2 == 0)
     {
      Print("Error: Standard deviation is zero in MathCorrelation.");
      return 0.0;
     }
   
   return cov / (std1 * std2);
  }

//+------------------------------------------------------------------+
//| Augmented Dickey-Fuller (ADF) test for stationarity             |
//| Returns p-value (approximate via t-stat comparison; simplified) |
//+------------------------------------------------------------------+
double ADFTest(double &spread[])
  {
   int n = ArraySize(spread);
   if(n < 10) return 1.0; // Insufficient data
   
   // Compute differences: Δy_t = y_t - y_{t-1}
   double delta[];
   ArrayResize(delta, n-1);
   for(int i = 0; i < n-1; i++)
     {
      delta[i] = spread[i] - spread[i+1]; // Note: series reversed if needed
     }
   
   // Simple ADF with 1 lag (rho test: regress Δy on y_{t-1} and Δy_{t-1})
   double y_lag[], delta_lag[];
   ArrayResize(y_lag, n-1);
   ArrayResize(delta_lag, n-2);
   for(int i = 0; i < n-1; i++)
     {
      y_lag[i] = spread[i+1]; // y_{t-1}
      if(i < n-2) delta_lag[i] = delta[i+1]; // Δy_{t-1}
     }
   
   // OLS for ADF: Δy_t = α + β y_{t-1} + γ Δy_{t-1} + ε (β = - (1 - rho))
   // Here, simplified to t-stat on β (critical values approx: -2.9 for 5% 1-sided)
   double cov_y_delta = MathCovariance(y_lag, delta);
   double var_y = MathVariance(y_lag);
   double beta = cov_y_delta / var_y; // Approx β
   double se_beta = MathSqrt(MathAbs(cov_y_delta / var_y)); // Rough SE
   double t_stat = beta / se_beta;
   
   // Approximate p-value (t-dist with df = n-2; use rough lookup or formula)
   // For simplicity, use rule: p ≈ 1 - norm.cdf(|t| + 2.9) but custom approx
   double p_approx = (t_stat > -2.9) ? 0.6 : (t_stat > -3.5 ? 0.1 : 0.01); // Coarse but functional
   
   return p_approx;
  }

//+------------------------------------------------------------------+
//| Calculate central tendency (mean or median)                      |
//+------------------------------------------------------------------+
double CalculateCentralTendency(double &array[])
  {
   switch(CT_Type)
     {
      case CT_Mean: return MathMean(array);
      case CT_Median: return MathMedian(array);
      default: return MathMean(array);
     }
  }

//+------------------------------------------------------------------+
//| Calculate dispersion (SD or MAD)                                 |
//+------------------------------------------------------------------+
double CalculateDispersion(double &array[])
  {
   switch(DT_Type)
     {
      case DT_StdDev: return MathStandardDeviation(array);
      case DT_MAD:
        {
         double med = MathMedian(array);
         double devs[];
         int n = ArraySize(array);
         ArrayResize(devs, n);
         for(int i = 0; i < n; i++)
           {
            devs[i] = MathAbs(array[i] - med);
           }
         double mad = MathMedian(devs);
         return mad * 1.4826; // Scale to SD equivalence for normal dist
        }
      default: return MathStandardDeviation(array);
     }
  }

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Trade.SetExpertMagicNumber(MagicNumber);
   
   // Validate symbol inputs
   if(StringLen(SymbolA) == 0 || StringLen(SymbolB) == 0)
     {
      Print("Error: SymbolA and SymbolB must be specified in input parameters.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   
   // Verify symbols exist
   if(!SymbolSelect(SymbolA, true) || !SymbolSelect(SymbolB, true))
     {
      Print("Error: One or both symbols (", SymbolA, ", ", SymbolB, ") not available in Market Watch.");
      Print("Please check your broker's Market Watch for valid symbol names (e.g., XAUUSD, XAGUSD, AUDUSD, NZDUSD).");
      return(INIT_FAILED);
     }
   
   // Verify symbol data availability
   double testPriceA, testPriceB;
   if(!SymbolInfoDouble(SymbolA, SYMBOL_BID, testPriceA) || !SymbolInfoDouble(SymbolB, SYMBOL_BID, testPriceB))
     {
      Print("Error: Cannot retrieve price data for ", SymbolA, " or ", SymbolB, ".");
      return(INIT_FAILED);
     }
   
   // Set initial dynamic lookback
   DynamicLookback = LookbackPeriod;
   
   // Calculate initial hedge ratio
   if(!CalculateHedgeRatio())
     {
      Print("Failed to calculate hedge ratio. Check data availability for ", SymbolA, " and ", SymbolB, ".");
      Print("Suggestion: Try pairs like XAUUSD-XAGUSD or AUDUSD-NZDUSD, or lower MinCorrelation for testing.");
      return(INIT_FAILED);
     }
   
   LastHedgeTime = iTime(SymbolA, Timeframe, 0);
   Print("Hedge Ratio (Beta): ", StringFormat("%.4f", Beta));
   
   // Initialize EWMA with full calculation
   if(UseEWMA)
     {
      UpdateEWMA(true); // Force full init
      EW_Initialized = true;
     }
   
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // HFT mode: run on every tick, no IsNewBar check

   // Check if market is open for both symbols
   if(!IsMarketOpen()) return;
   
   // Rolling hedge update on new bar if enabled
   if(UseRollingHedge)
     {
      datetime currentBarTime = iTime(SymbolA, Timeframe, 0);
      if(currentBarTime != LastHedgeTime)
        {
         int barsSinceLast = iBarShift(SymbolA, Timeframe, LastHedgeTime);
         if(barsSinceLast >= HedgeUpdateBars)
           {
            if(CalculateHedgeRatio())
              {
               Print("Rolling hedge updated: Beta = ", StringFormat("%.4f", Beta));
               LastHedgeTime = currentBarTime;
              }
           }
        }
     }
   
   // Get historical closes
   double closesA[], closesB[];
   ArraySetAsSeries(closesA, true);
   ArraySetAsSeries(closesB, true);
   int barsNeeded = DynamicLookback + 1;
   
   if(CopyClose(SymbolA, Timeframe, 0, barsNeeded, closesA) < barsNeeded ||
      CopyClose(SymbolB, Timeframe, 0, barsNeeded, closesB) < barsNeeded)
     {
      Print("Insufficient historical data for ", SymbolA, " or ", SymbolB);
      return;
     }
   
   // Calculate historical spreads (excluding current bar)
   double spreads[];
   ArrayResize(spreads, DynamicLookback);
   ArraySetAsSeries(spreads, true);
   for(int i = 0; i < DynamicLookback; i++)
     {
      spreads[i] = closesA[i + 1] - Beta * closesB[i + 1];
     }
   
   // Update mu and sigma (EWMA or full)
   if(UseEWMA)
     {
      UpdateEWMA(false); // Incremental update
     }
   else
     {
      double mu = CalculateCentralTendency(spreads);
      double sigma = CalculateDispersion(spreads);
      if(sigma == 0)
        {
         Print("Error: Spread dispersion is zero.");
         return;
        }
      EW_Mu = mu; // Sync for consistency
      EW_Sigma = sigma;
     }
   double mu = EW_Mu;
   double sigma = EW_Sigma;
   if(sigma == 0)
     {
      Print("Error: Spread dispersion is zero.");
      return;
     }
   
   // Current spread using Bid/Ask for HFT
   double bidA = SymbolInfoDouble(SymbolA, SYMBOL_BID);
   double askA = SymbolInfoDouble(SymbolA, SYMBOL_ASK);
   double bidB = SymbolInfoDouble(SymbolB, SYMBOL_BID);
   double askB = SymbolInfoDouble(SymbolB, SYMBOL_ASK);
   
   // For long spread: use askA (buy A) and bidB (sell B)
   double longSpread = askA - Beta * bidB;
   // For short spread: use bidA (sell A) and askB (buy B)
   double shortSpread = bidA - Beta * askB;
   
   // Use average or mid for Z-Score calculation
   double currentSpread = (longSpread + shortSpread) / 2;
   double zScore = (currentSpread - mu) / sigma;
   Print("Z-Score: ", StringFormat("%.4f", zScore));
   
   // Get current direction
   int direction = GetPositionDirection();
   
   if(direction == 0)
     {
      // Check for entry
      if(zScore <= -EntryZScore)
        {
         // Long the spread: Buy A, Sell B
         double lotsA = CalculateLots(sigma);
         double lotsB = Beta * lotsA;
         OpenLongSpread(lotsA, lotsB, zScore);
        }
      else if(zScore >= EntryZScore)
        {
         // Short the spread: Sell A, Buy B
         double lotsA = CalculateLots(sigma);
         double lotsB = Beta * lotsA;
         OpenShortSpread(lotsA, lotsB, zScore);
        }
     }
   else
     {
      // Check for exit or stop
      bool shouldClose = false;
      double currentProfit = CalculatePairProfit();
      
      if(direction == 1) // Long spread
        {
         if(SL_Type == SL_ZScore)
           {
            double sl_z = EntryZ - (StopZScore - EntryZScore);
            double tp_z;
            if(TP_Type == TP_Multiple)
              tp_z = EntryZ + RiskRewardRatio * (StopZScore - EntryZScore);
            else if(TP_Type == TP_ZScore)
              tp_z = TakeProfitZScore; // e.g., 0.0 for mean reversion
            if(zScore <= sl_z || zScore >= tp_z) shouldClose = true;
           }
         else if(SL_Type == SL_Percent)
           {
            double sl_amount = - (StopLossPercent / 100.0) * EntryEquity;
            double tp_amount = -sl_amount * RiskRewardRatio;
            if(currentProfit <= sl_amount || currentProfit >= tp_amount) shouldClose = true;
           }
        }
      else if(direction == -1) // Short spread
        {
         if(SL_Type == SL_ZScore)
           {
            double sl_z = EntryZ + (StopZScore - EntryZScore);
            double tp_z;
            if(TP_Type == TP_Multiple)
              tp_z = EntryZ - RiskRewardRatio * (StopZScore - EntryZScore);
            else if(TP_Type == TP_ZScore)
              tp_z = TakeProfitZScore; // e.g., 0.0 for mean reversion
            if(zScore >= sl_z || zScore <= tp_z) shouldClose = true;
           }
         else if(SL_Type == SL_Percent)
           {
            double sl_amount = - (StopLossPercent / 100.0) * EntryEquity;
            double tp_amount = -sl_amount * RiskRewardRatio;
            if(currentProfit <= sl_amount || currentProfit >= tp_amount) shouldClose = true;
           }
        }
      
      if(shouldClose)
        {
         CloseAllPositions();
        }
     }
  }

//+------------------------------------------------------------------+
//| Update EWMA for mu and variance (sigma = sqrt(var))             |
//+------------------------------------------------------------------+
void UpdateEWMA(bool forceFull)
  {
   if(!UseEWMA) return;
   
   // Get current spread for update (use latest historical as proxy if no new)
   double closesA[1], closesB[1];
   if(CopyClose(SymbolA, Timeframe, 0, 1, closesA) < 1 || CopyClose(SymbolB, Timeframe, 0, 1, closesB) < 1)
      return;
   double currentSpreadHist = closesA[0] - Beta * closesB[0];
   
   if(forceFull || !EW_Initialized)
     {
      // Full init
      double spreads[];
      ArrayResize(spreads, DynamicLookback);
      // Assume spreads filled as in OnTick (reuse logic if needed)
      // For simplicity, compute from closes (full would require full array; approximate here)
      double closesA_full[], closesB_full[];
      ArraySetAsSeries(closesA_full, true);
      ArraySetAsSeries(closesB_full, true);
      if(CopyClose(SymbolA, Timeframe, 0, DynamicLookback + 1, closesA_full) >= DynamicLookback + 1 &&
         CopyClose(SymbolB, Timeframe, 0, DynamicLookback + 1, closesB_full) >= DynamicLookback + 1)
        {
         for(int i = 0; i < DynamicLookback; i++)
           {
            spreads[i] = closesA_full[i + 1] - Beta * closesB_full[i + 1];
           }
         EW_Mu = CalculateCentralTendency(spreads);
         double temp_var = 0.0;
         for(int i = 0; i < DynamicLookback; i++)
           {
            temp_var += MathPow(spreads[i] - EW_Mu, 2);
           }
         EW_Var = temp_var / DynamicLookback;
         EW_Sigma = MathSqrt(EW_Var);
        }
      EW_Initialized = true;
      return;
     }
   
   // Incremental EWMA update
   double alpha = Alpha;
   double old_mu = EW_Mu;
   EW_Mu = alpha * currentSpreadHist + (1 - alpha) * old_mu;
   
   // EW Variance update
   double sq_diff = MathPow(currentSpreadHist - old_mu, 2);
   EW_Var = alpha * sq_diff + (1 - alpha) * EW_Var;
   EW_Sigma = MathSqrt(EW_Var);
  }

//+------------------------------------------------------------------+
//| Calculate hedge ratio (Beta) with ADF adjustment                 |
//+------------------------------------------------------------------+
bool CalculateHedgeRatio()
  {
   double closesA[], closesB[];
   ArraySetAsSeries(closesA, true);
   ArraySetAsSeries(closesB, true);
   
   if(CopyClose(SymbolA, Timeframe, 0, RegressionPeriod, closesA) < RegressionPeriod ||
      CopyClose(SymbolB, Timeframe, 0, RegressionPeriod, closesB) < RegressionPeriod)
     {
      Print("Error: Insufficient historical data for ", SymbolA, " or ", SymbolB);
      return false;
     }
   
   // Calculate daily returns for correlation
   int n = RegressionPeriod - 1;
   double returnsA[], returnsB[];
   ArrayResize(returnsA, n);
   ArrayResize(returnsB, n);
   ArraySetAsSeries(returnsA, true);
   ArraySetAsSeries(returnsB, true);
   for(int i = 0; i < n; i++)
     {
      returnsA[i] = (closesA[i] - closesA[i + 1]) / closesA[i + 1];
      returnsB[i] = (closesB[i] - closesB[i + 1]) / closesB[i + 1];
     }
   
   // Check correlation (simple proxy for cointegration check)
   double correlation = MathCorrelation(returnsA, returnsB);
   Print("Pair Correlation: ", StringFormat("%.4f", correlation));
   if(correlation < MinCorrelation && !BypassCorrelationCheck)
     {
      Print("Correlation (", StringFormat("%.4f", correlation), ") below threshold (", MinCorrelation, "). Not trading this pair.");
      return false;
     }
   
   Print("Correlation check passed.");
   
   // Calculate Beta: cov(A,B) / var(B) on prices
   double cov = MathCovariance(closesA, closesB);
   double varB = MathVariance(closesB);
   if(varB == 0)
     {
      Print("Error: Variance of ", SymbolB, " is zero. Cannot calculate Beta.");
      return false;
     }
   
   Beta = cov / varB;
   
   // ADF Test on spreads if enabled
   if(UseADFTest)
     {
      double spreads_reg[];
      ArrayResize(spreads_reg, RegressionPeriod);
      for(int i = 0; i < RegressionPeriod; i++)
        {
         spreads_reg[i] = closesA[i] - Beta * closesB[i];
        }
      double p_value = ADFTest(spreads_reg);
      Print("ADF p-value on spreads: ", StringFormat("%.4f", p_value));
      
      // Adjust dynamic lookback if non-stationary
      int temp_lookback = LookbackPeriod;
      while(p_value > ADF_PThreshold && temp_lookback >= MinLookback)
        {
         temp_lookback /= 2;
         // Recompute spreads with shorter window
         ArrayResize(spreads_reg, temp_lookback);
         for(int i = 0; i < temp_lookback; i++)
           {
            spreads_reg[i] = closesA[i] - Beta * closesB[i];
           }
         p_value = ADFTest(spreads_reg);
         Print("Shortened lookback to ", temp_lookback, "; new p-value: ", StringFormat("%.4f", p_value));
        }
      DynamicLookback = MathMax(MinLookback, temp_lookback);
      if(p_value > ADF_PThreshold)
        {
         Print("Warning: Spread remains non-stationary even at min lookback. Trading may be risky.");
        }
     }
   else
     {
      DynamicLookback = LookbackPeriod;
     }
   
   Print("Calculated Beta: ", StringFormat("%.4f", Beta), "; Dynamic Lookback: ", DynamicLookback);
   
   return true;
  }

//+------------------------------------------------------------------+
//| Calculate lots for Asset A based on risk                         |
//+------------------------------------------------------------------+
double CalculateLots(double sigma)
  {
   if(sigma <= 0)
     {
      Print("Error: Sigma is zero or negative. Cannot calculate lots.");
      return 0.0;
     }
   
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = (RiskPercent / 100.0) * accountBalance;
   if(SL_Type == SL_Percent)
     {
      riskAmount = (StopLossPercent / 100.0) * accountBalance;
     }
   
   double adverseMove = 2.0 * sigma; // From entry to stop (e.g., 2 to 4 or -2 to -4)
   
   double tick_size = SymbolInfoDouble(SymbolA, SYMBOL_POINT);
   double tick_value = SymbolInfoDouble(SymbolA, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size == 0 || tick_value == 0)
     {
      Print("Error: Invalid tick size or value for ", SymbolA);
      return 0.0;
     }
   double dollar_per_unit_per_lot = tick_value / tick_size;
   
   double lotsA = riskAmount / (adverseMove * dollar_per_unit_per_lot);
   
   // Normalize to broker's lot step
   double lotStep = SymbolInfoDouble(SymbolA, SYMBOL_VOLUME_STEP);
   if(lotStep > 0)
      lotsA = MathFloor(lotsA / lotStep) * lotStep;
   
   // Ensure within min/max lot sizes
   double minLot = SymbolInfoDouble(SymbolA, SYMBOL_VOLUME_MIN);
   double maxLot = MathMin(SymbolInfoDouble(SymbolA, SYMBOL_VOLUME_MAX), MaxLots); // Cap at MaxLots
   lotsA = MathMax(minLot, MathMin(maxLot, lotsA));
   
   return lotsA;
  }

//+------------------------------------------------------------------+
//| Get current position direction                                   |
//+------------------------------------------------------------------+
int GetPositionDirection()
  {
   bool hasPosA = false, hasPosB = false;
   long typeA = -1, typeB = -1;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i))
        {
         if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
         
         string posSymbol = PositionGetString(POSITION_SYMBOL);
         if(posSymbol == SymbolA)
           {
            hasPosA = true;
            typeA = PositionGetInteger(POSITION_TYPE);
           }
         else if(posSymbol == SymbolB)
           {
            hasPosB = true;
            typeB = PositionGetInteger(POSITION_TYPE);
           }
        }
     }
   
   if(!hasPosA || !hasPosB) return 0;
   
   if(typeA == POSITION_TYPE_BUY && typeB == POSITION_TYPE_SELL) return 1; // Long spread
   if(typeA == POSITION_TYPE_SELL && typeB == POSITION_TYPE_BUY) return -1; // Short spread
   
   return 0; // Mismatched, treat as none
  }

//+------------------------------------------------------------------+
//| Calculate total profit for the pair                              |
//+------------------------------------------------------------------+
double CalculatePairProfit()
  {
   double profit = 0.0;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i))
        {
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            (PositionGetString(POSITION_SYMBOL) == SymbolA || PositionGetString(POSITION_SYMBOL) == SymbolB))
           {
            profit += PositionGetDouble(POSITION_PROFIT);
           }
        }
     }
   
   return profit;
  }

//+------------------------------------------------------------------+
//| Open long spread position                                        |
//+------------------------------------------------------------------+
void OpenLongSpread(double lotsA, double lotsB, double zScore)
  {
   if(lotsA <= 0 || lotsB <= 0)
     {
      Print("Error: Invalid lot sizes for long spread (A: ", lotsA, ", B: ", lotsB, ")");
      return;
     }
   
   // Adjust lotB to broker's lot step for SymbolB
   double lotStepB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_STEP);
   if(lotStepB > 0)
      lotsB = MathFloor(lotsB / lotStepB) * lotStepB;
   
   double minLotB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MIN);
   double maxLotB = MathMin(SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MAX), MaxLots); // Cap at MaxLots
   lotsB = MathMax(minLotB, MathMin(maxLotB, lotsB));
   
   // Buy A
   if(!Trade.PositionOpen(SymbolA, ORDER_TYPE_BUY, lotsA, 0, 0, 0, "Pairs Long Spread"))
      Print("Error opening BUY position for ", SymbolA, ": ", GetLastError());
   
   // Sell B
   if(!Trade.PositionOpen(SymbolB, ORDER_TYPE_SELL, lotsB, 0, 0, 0, "Pairs Long Spread"))
      Print("Error opening SELL position for ", SymbolB, ": ", GetLastError());
   
   // Store entry values
   EntryZ = zScore;
   EntryEquity = AccountInfoDouble(ACCOUNT_EQUITY);
  }

//+------------------------------------------------------------------+
//| Open short spread position                                       |
//+------------------------------------------------------------------+
void OpenShortSpread(double lotsA, double lotsB, double zScore)
  {
   if(lotsA <= 0 || lotsB <= 0)
     {
      Print("Error: Invalid lot sizes for short spread (A: ", lotsA, ", B: ", lotsB, ")");
      return;
     }
   
   // Adjust lotB to broker's lot step for SymbolB
   double lotStepB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_STEP);
   if(lotStepB > 0)
      lotsB = MathFloor(lotsB / lotStepB) * lotStepB;
   
   double minLotB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MIN);
   double maxLotB = MathMin(SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MAX), MaxLots); // Cap at MaxLots
   lotsB = MathMax(minLotB, MathMin(maxLotB, lotsB));
   
   // Sell A
   if(!Trade.PositionOpen(SymbolA, ORDER_TYPE_SELL, lotsA, 0, 0, 0, "Pairs Short Spread"))
      Print("Error opening SELL position for ", SymbolA, ": ", GetLastError());
   
   // Buy B
   if(!Trade.PositionOpen(SymbolB, ORDER_TYPE_BUY, lotsB, 0, 0, 0, "Pairs Short Spread"))
      Print("Error opening BUY position for ", SymbolB, ": ", GetLastError());
   
   // Store entry values
   EntryZ = zScore;
   EntryEquity = AccountInfoDouble(ACCOUNT_EQUITY);
  }

//+------------------------------------------------------------------+
//| Close all positions for the pair                                 |
//+------------------------------------------------------------------+
void CloseAllPositions()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i))
        {
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            (PositionGetString(POSITION_SYMBOL) == SymbolA || PositionGetString(POSITION_SYMBOL) == SymbolB))
           {
            if(!Trade.PositionClose(PositionGetInteger(POSITION_TICKET)))
               Print("Error closing position for ", PositionGetString(POSITION_SYMBOL), ": ", GetLastError());
           }
        }
     }
  }

//+------------------------------------------------------------------+