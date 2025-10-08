//+------------------------------------------------------------------+
//|                                             REHOBOAM-v2.mq5      |
//|                        Copyright 2025, Stephen Njai               |
//|                                             https://github.com/SteveNjai |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Stephen Njai"
#property link      "https://github.com/SteveNjai"
#property version   "2.0"
#property strict
#property description "Pairs Trading EA utilizing correlation between GBPUSD/EURUSD."
#property description "Uses 2-second resampled tick data. Hedge ratio calculated with 3600 2s bars."
#property description "Position sizing risks 1% of balance based on 2*sigma for stop at +/-2.5 Z."
#property description "v2: Modified for HFT, processes ticks resampled to 2s, reads optimal EntryZScore."

#include <Math\Stat\Math.mqh>
#include <Trade\Trade.mqh>
#include <Files\FileTxt.mqh>

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

// Inputs
input string SymbolA = "GBPUSD";    // Symbol for Asset A
input string SymbolB = "EURUSD";    // Symbol for Asset B
input int LookbackPeriod = 600;     // Lookback for mean and std dev of spread (2s bars, ~20 minutes)
input int RegressionPeriod = 3600;  // Lookback for hedge ratio calculation (2s bars, ~2 hours)
input double DefaultEntryZScore = 2.0; // Default entry threshold for |Z-Score|
input double StopZScore = 4.8;      // Stop-loss threshold for |Z-Score|
input double TakeProfitZScore = 0.0; // Take profit Z-score threshold
input double RiskPercent = 1.0;     // Risk % of account balance per trade
input double MinCorrelation = 0.2;  // Minimum correlation to allow trading
input bool BypassCorrelationCheck = false; // Bypass correlation check
input bool BypassCointCheck = true;  // Bypass cointegration check
input long MagicNumber = 12345;     // Magic number for positions
input double RiskRewardRatio = 2.0; // Take profit multiple of stop loss
input StopLossType SL_Type = SL_ZScore; // Stop loss type
input TakeProfitType TP_Type = TP_Multiple; // Take profit type
input double StopLossPercent = 2.5; // Stop loss % of entry equity
input double MaxLots = 1.0;         // Maximum lot size per trade

// Globals
double Beta = 0.0;
CTrade Trade;
double EntryZ = 0.0;
double EntryEquity = 0.0;
double EntryZScore = DefaultEntryZScore;
datetime last_timestamp = 0;
double last_bidA = 0, last_askA = 0, last_bidB = 0, last_askB = 0;
double spreads[];
int spread_idx = 0;

//+------------------------------------------------------------------+
//| Read optimal Z-score from file                                   |
//+------------------------------------------------------------------+
void UpdateEntryZScore()
{
   CFileTxt file;
   if(file.Open("optimal_zscore.json", FILE_READ | FILE_COMMON))
   {
      string json = file.ReadString();
      file.Close();
      int pos = StringFind(json, "\"optimal_z\":");
      if(pos != -1)
      {
         string value = StringSubstr(json, pos + 12, StringLen(json) - pos - 13);
         double new_z = StringToDouble(value);
         if(new_z >= 0 && new_z <= 8)
         {
            EntryZScore = new_z;
            Print("Updated EntryZScore to ", EntryZScore);
         }
         else
            Print("Error: Z-Score ", new_z, " out of range [0, 8]");
      }
      else
         Print("Error: Failed to parse JSON: ", json);
   }
   else
      Print("Error: Failed to open optimal_zscore.json, error code: ", GetLastError());
}

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
   if(StringLen(SymbolA) == 0 || StringLen(SymbolB) == 0)
     {
      Print("Error: SymbolA and SymbolB must be specified.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(!SymbolSelect(SymbolA, true) || !SymbolSelect(SymbolB, true))
     {
      Print("Error: Symbols (", SymbolA, ", ", SymbolB, ") not in Market Watch.");
      return(INIT_FAILED);
     }
   double testPriceA, testPriceB;
   if(!SymbolInfoDouble(SymbolA, SYMBOL_BID, testPriceA) || !SymbolInfoDouble(SymbolB, SYMBOL_BID, testPriceB))
     {
      Print("Error: Cannot retrieve price data for ", SymbolA, " or ", SymbolB, ".");
      return(INIT_FAILED);
     }
   if(!CalculateHedgeRatio())
     {
      Print("Failed to calculate hedge ratio.");
      return(INIT_FAILED);
     }
   Print("Hedge Ratio (Beta): ", StringFormat("%.4f", Beta));
   ArrayResize(spreads, LookbackPeriod);
   ArrayInitialize(spreads, 0);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsMarketOpen()) return;
   UpdateEntryZScore();

   // Get current tick data
   double bidA = SymbolInfoDouble(SymbolA, SYMBOL_BID);
   double askA = SymbolInfoDouble(SymbolA, SYMBOL_ASK);
   double bidB = SymbolInfoDouble(SymbolB, SYMBOL_BID);
   double askB = SymbolInfoDouble(SymbolB, SYMBOL_ASK);
   datetime current_time = TimeCurrent();

   // Check if 2 seconds have passed
   if(current_time >= last_timestamp + 2)
     {
      last_timestamp = current_time;
      last_bidA = bidA;
      last_askA = askA;
      last_bidB = bidB;
      last_askB = askB;

      // Update spread array
      double current_spread = (last_askA - Beta * last_bidB + last_bidA - Beta * last_askB) / 2;
      if(spread_idx < LookbackPeriod)
        {
         spreads[spread_idx] = current_spread;
         spread_idx++;
        }
      else
        {
         ArrayCopy(spreads, spreads, 0, 1, LookbackPeriod-1);
         spreads[LookbackPeriod-1] = current_spread;
        }

      // Calculate Z-score if enough data
      if(spread_idx >= LookbackPeriod)
        {
         double mu = MathMean(spreads);
         double sigma = MathStandardDeviation(spreads);
         if(sigma == 0)
           {
            Print("Error: Spread standard deviation is zero.");
            return;
           }
         double zScore = (current_spread - mu) / sigma;
         Print("Z-Score (2s): ", StringFormat("%.4f", zScore));

         int direction = GetPositionDirection();
         if(direction == 0)
           {
            if(zScore <= -EntryZScore)
              {
               double lotsA = CalculateLots(sigma);
               double lotsB = Beta * lotsA;
               OpenLongSpread(lotsA, lotsB, zScore);
              }
            else if(zScore >= EntryZScore)
              {
               double lotsA = CalculateLots(sigma);
               double lotsB = Beta * lotsA;
               OpenShortSpread(lotsA, lotsB, zScore);
              }
           }
         else
           {
            bool shouldClose = false;
            double currentProfit = CalculatePairProfit();
            if(direction == 1)
              {
               if(SL_Type == SL_ZScore)
                 {
                  double sl_z = EntryZ - (StopZScore - EntryZScore);
                  double tp_z = (TP_Type == TP_Multiple) ? EntryZ + RiskRewardRatio * (StopZScore - EntryZScore) : TakeProfitZScore;
                  if(zScore <= sl_z || zScore >= tp_z) shouldClose = true;
                 }
               else
                 {
                  double sl_amount = -(StopLossPercent / 100.0) * EntryEquity;
                  double tp_amount = -sl_amount * RiskRewardRatio;
                  if(currentProfit <= sl_amount || currentProfit >= tp_amount) shouldClose = true;
                 }
              }
            else if(direction == -1)
              {
               if(SL_Type == SL_ZScore)
                 {
                  double sl_z = EntryZ + (StopZScore - EntryZScore);
                  double tp_z = (TP_Type == TP_Multiple) ? EntryZ - RiskRewardRatio * (StopZScore - EntryZScore) : TakeProfitZScore;
                  if(zScore >= sl_z || zScore <= tp_z) shouldClose = true;
                 }
               else
                 {
                  double sl_amount = -(StopLossPercent / 100.0) * EntryEquity;
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
   if(CopyClose(SymbolA, PERIOD_M1, 0, RegressionPeriod/30, closesA) < RegressionPeriod/30 ||
      CopyClose(SymbolB, PERIOD_M1, 0, RegressionPeriod/30, closesB) < RegressionPeriod/30)
     {
      Print("Error: Insufficient historical data for ", SymbolA, " or ", SymbolB);
      return false;
     }
   int n = ArraySize(closesA) - 1;
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
   double correlation = MathCorrelation(returnsA, returnsB);
   Print("Pair Correlation: ", StringFormat("%.4f", correlation));
   if(correlation < MinCorrelation && !BypassCorrelationCheck)
     {
      Print("Correlation (", StringFormat("%.4f", correlation), ") below threshold (", MinCorrelation, ").");
      return false;
     }
   double cov = MathCovariance(closesA, closesB);
   double varB = MathVariance(closesB);
   if(varB == 0)
     {
      Print("Error: Variance of ", SymbolB, " is zero.");
      return false;
     }
   Beta = cov / varB;
   Print("Calculated Beta: ", StringFormat("%.4f", Beta));
   return true;
  }

//+------------------------------------------------------------------+
//| Calculate lots for Asset A based on risk                         |
//+------------------------------------------------------------------+
double CalculateLots(double sigma)
  {
   if(sigma <= 0)
     {
      Print("Error: Sigma is zero or negative.");
      return 0.0;
     }
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = (RiskPercent / 100.0) * accountBalance;
   if(SL_Type == SL_Percent)
     {
      riskAmount = (StopLossPercent / 100.0) * accountBalance;
     }
   double adverseMove = 2.0 * sigma;
   double tick_size = SymbolInfoDouble(SymbolA, SYMBOL_POINT);
   double tick_value = SymbolInfoDouble(SymbolA, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size == 0 || tick_value == 0)
     {
      Print("Error: Invalid tick size or value for ", SymbolA);
      return 0.0;
     }
   double dollar_per_unit_per_lot = tick_value / tick_size;
   double lotsA = riskAmount / (adverseMove * dollar_per_unit_per_lot);
   double lotStep = SymbolInfoDouble(SymbolA, SYMBOL_VOLUME_STEP);
   if(lotStep > 0)
      lotsA = MathFloor(lotsA / lotStep) * lotStep;
   double minLot = SymbolInfoDouble(SymbolA, SYMBOL_VOLUME_MIN);
   double maxLot = MathMin(SymbolInfoDouble(SymbolA, SYMBOL_VOLUME_MAX), MaxLots);
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
   if(typeA == POSITION_TYPE_BUY && typeB == POSITION_TYPE_SELL) return 1;
   if(typeA == POSITION_TYPE_SELL && typeB == POSITION_TYPE_BUY) return -1;
   return 0;
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
   double lotStepB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_STEP);
   if(lotStepB > 0)
      lotsB = MathFloor(lotsB / lotStepB) * lotStepB;
   double minLotB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MIN);
   double maxLotB = MathMin(SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MAX), MaxLots);
   lotsB = MathMax(minLotB, MathMin(maxLotB, lotsB));
   if(!Trade.PositionOpen(SymbolA, ORDER_TYPE_BUY, lotsA, 0, 0, 0, "Pairs Long Spread"))
      Print("Error opening BUY position for ", SymbolA, ": ", GetLastError());
   if(!Trade.PositionOpen(SymbolB, ORDER_TYPE_SELL, lotsB, 0, 0, 0, "Pairs Long Spread"))
      Print("Error opening SELL position for ", SymbolB, ": ", GetLastError());
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
   double lotStepB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_STEP);
   if(lotStepB > 0)
      lotsB = MathFloor(lotsB / lotStepB) * lotStepB;
   double minLotB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MIN);
   double maxLotB = MathMin(SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MAX), MaxLots);
   lotsB = MathMax(minLotB, MathMin(maxLotB, lotsB));
   if(!Trade.PositionOpen(SymbolA, ORDER_TYPE_SELL, lotsA, 0, 0, 0, "Pairs Short Spread"))
      Print("Error opening SELL position for ", SymbolA, ": ", GetLastError());
   if(!Trade.PositionOpen(SymbolB, ORDER_TYPE_BUY, lotsB, 0, 0, 0, "Pairs Short Spread"))
      Print("Error opening BUY position for ", SymbolB, ": ", GetLastError());
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