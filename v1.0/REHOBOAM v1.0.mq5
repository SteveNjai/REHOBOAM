//+------------------------------------------------------------------+
//|                                             REHOBOAM-v1.1.mq5    |
//|                        Copyright 2025, Stephen Njai               |
//|                                             https://github.com/SteveNjai |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Stephen Njai"
#property link      "https://github.com/SteveNjai"
#property version   "1.1"
#property strict
#property description "Pairs Trading EA utilizing correlation between pairs for user-specified symbols."
#property description "Uses daily or 1H timeframe. Hedge ratio calculated on init with 252 days."
#property description "Position sizing risks 1% of balance based on 2*sigma for stop at +/-4 Z or percentage based stop loss."
#property description "Uses correlation as a proxy for cointegration."

#include <Math\Stat\Math.mqh>
#include <Trade\Trade.mqh>

// Stop Loss Types
enum StopLossType
  {
   SL_ZScore = 0,   // Z-Score based stop loss
   SL_Percent = 1   // Percentage based stop loss
  };

// Inputs
input string SymbolA = "";           // Symbol for Asset A (e.g., XAUUSD, leave empty to input manually)
input string SymbolB = "";           // Symbol for Asset B (e.g., XAGUSD, leave empty to input manually)
input ENUM_TIMEFRAMES Timeframe = PERIOD_D1; // Timeframe (Daily recommended)
input int LookbackPeriod = 20;      // Lookback for mean and std dev of spread
input int RegressionPeriod = 252;   // Lookback for hedge ratio calculation
input double EntryZScore = 1.8;     // Entry threshold for |Z-Score|
input double StopZScore = 4.0;      // Stop-loss threshold for |Z-Score| (used if SL_ZScore)
input double RiskPercent = 1.0;     // Risk % of account balance per trade (for sizing)
input double MinCorrelation = 0.5;  // Minimum correlation to allow trading
input bool BypassCorrelationCheck = false; // Bypass correlation check for testing (use with caution)
input long MagicNumber = 12345;     // Magic number for positions
input double RiskRewardRatio = 2.0; // Take profit multiple of stop loss
input StopLossType SL_Type = SL_Percent; // Stop loss type
input double StopLossPercent = 1.0; // Stop loss % of entry equity (used if SL_Percent)
input double MaxLots = 2.0;         // Maximum lot size per trade to prevent margin issues

// Globals
double Beta = 0.0;
CTrade Trade;
datetime LastBarTime = 0;
double EntryZ = 0.0; // Entry Z-Score
double EntryEquity = 0.0; // Equity at entry

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
   
   // Calculate hedge ratio
   if(!CalculateHedgeRatio())
     {
      Print("Failed to calculate hedge ratio. Check data availability for ", SymbolA, " and ", SymbolB, ".");
      Print("Suggestion: Try pairs like XAUUSD-XAGUSD or AUDUSD-NZDUSD, or lower MinCorrelation for testing.");
      return(INIT_FAILED);
     }
   
   Print("Hedge Ratio (Beta): ", StringFormat("%.4f", Beta));
   
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewBar()) return;
   
   // Check if market is open for both symbols
   if(!IsMarketOpen()) return;
   
   // Get historical closes
   double closesA[], closesB[];
   ArraySetAsSeries(closesA, true);
   ArraySetAsSeries(closesB, true);
   int barsNeeded = LookbackPeriod + 1;
   
   if(CopyClose(SymbolA, Timeframe, 0, barsNeeded, closesA) < barsNeeded ||
      CopyClose(SymbolB, Timeframe, 0, barsNeeded, closesB) < barsNeeded)
     {
      Print("Insufficient historical data for ", SymbolA, " or ", SymbolB);
      return;
     }
   
   // Calculate historical spreads (excluding current bar)
   double spreads[];
   ArrayResize(spreads, LookbackPeriod);
   ArraySetAsSeries(spreads, true);
   for(int i = 0; i < LookbackPeriod; i++)
     {
      spreads[i] = closesA[i + 1] - Beta * closesB[i + 1];
     }
   
   // Mean and std dev of spread
   double mu = MathMean(spreads);
   double sigma = MathStandardDeviation(spreads);
   if(sigma == 0)
     {
      Print("Error: Spread standard deviation is zero.");
      return;
     }
   
   // Current spread and Z-Score
   double currentSpread = closesA[0] - Beta * closesB[0];
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
            double tp_z = EntryZ + RiskRewardRatio * (StopZScore - EntryZScore);
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
            double tp_z = EntryZ - RiskRewardRatio * (StopZScore - EntryZScore);
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
//| Calculate hedge ratio (Beta)                                     |
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
   
   // Calculate Beta: cov(A,B) / var(B) on prices
   double cov = MathCovariance(closesA, closesB);
   double varB = MathVariance(closesB);
   if(varB == 0)
     {
      Print("Error: Variance of ", SymbolB, " is zero. Cannot calculate Beta.");
      return false;
     }
   
   Beta = cov / varB;
   Print("Calculated Beta: ", StringFormat("%.4f", Beta));
   
   return true;
  }

//+------------------------------------------------------------------+
//| Check if new bar has opened                                      |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime currentBarTime = iTime(SymbolA, Timeframe, 0);
   if(currentBarTime > LastBarTime)
     {
      LastBarTime = currentBarTime;
      return true;
     }
   return false;
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