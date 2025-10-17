//+------------------------------------------------------------------+
//|                                             REHOBOAM-v4.5.mq5    |
//|                        Copyright 2025, Stephen Njai               |
//|                                             https://github.com/SteveNjai |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Stephen Njai"
#property link      "https://github.com/SteveNjai"
#property version   "4.5"
#property strict
#property description "Pairs Trading EA utilizing correlation between GBPUSD/EURUSD."
#property description "Uses M1 bars. Hedge ratio calculated with 234 M1 bars."
#property description "Position sizing risks 1% of balance based on 2*sigma for stop at +/-4.8 Z."
#property description "v4: Integrated Z-score optimization from ORACLE-SPREAD, runs simulation in OnInit and every SimIntervalMinutes."
#property description "v4.1: Added MinEntryZScore to prevent trades below a minimum Z-score threshold, even if optimal."
#property description "v4.2: Uses M1 data exclusively, removed tick data and resampling."
#property description "v4.3: Handles negative Beta by using absolute value for lot sizing and adjusting SymbolB trade direction."
#property description "v4.4: Pre-loads historical spreads in OnInit for immediate runtime Z-score calculations and trade checks."
#property description "Prints current Z-score every minute (new M1 bar)."
#property description "v4.5: Added MaxOpenPositions to limit open spreads, detects and closes partial positions."
#property description "Saves results and logs to Common Files with symbol prefix (e.g., GBPUSD-EURUSD-rehoboam-v4-results.txt)."
#property description "Logging to file can be disabled with EnableLogging=false to improve performance."
#property description "Logs only critical events (init, errors, simulation results, trades) to keep file size small."
#property description "DynamicEntryZScore enables/disables Monte Carlo simulation for optimal Z-score; if false, uses DefaultEntryZScore."

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
input int LookbackPeriod = 160;     // Lookback for mean and std dev of spread (M1 bars)
input int RegressionPeriod = 350;   // Lookback for hedge ratio calculation (M1 bars)
input double DefaultEntryZScore = 2.0; // Default entry threshold for |Z-Score|
input double MinEntryZScore = 2.0;  // Minimum Z-score threshold for trade entry
input double StopZScore = 7.2;      // Stop-loss threshold for |Z-Score|
input double TakeProfitZScore = 2.0; // Take profit Z-score threshold
input double RiskPercent = 1.0;     // Risk % of account balance per trade
input double MinCorrelation = 0.2;  // Minimum correlation to allow trading
input bool BypassCorrelationCheck = false; // Bypass correlation check
input bool BypassCointCheck = true;  // Bypass cointegration check
input long MagicNumber = 1099;      // Magic number for positions
input double RiskRewardRatio = 2.0; // Take profit multiple of stop loss
input StopLossType SL_Type = SL_Percent; // Stop loss type
input TakeProfitType TP_Type = TP_Multiple; // Take profit type
input double StopLossPercent = 1.5; // Stop loss % of entry equity
input double MaxLots = 1.0;         // Maximum lot size per trade
input int SimNPaths = 100;          // Number of simulation paths
input int SimNSteps = 60;           // Number of steps per path (1hr at 1m)
input double ZScoreMax = 8.0;       // Maximum Z-score for simulation
input double ZScoreStep = 0.2;      // Z-score increment for testing
input double SimIntervalMinutes = 30.0; // Simulation interval (minutes)
input double LotSize = 0.1;         // Lot size for simulation
input double PipValue = 10.0;       // USD per pip for 1 lot
input double PipSize = 0.0001;      // Pip size for calculations
input bool EnableLogging = true;    // Enable logging to file (disable for faster backtests)
input bool DynamicEntryZScore = true; // Enable dynamic Z-score via Monte Carlo simulation
input int MaxOpenPositions = 1;     // Maximum number of open spreads

// Globals
double Beta = 0.0;
CTrade Trade;
double EntryZ = 0.0;
double EntryEquity = 0.0;
double EntryZScore = DefaultEntryZScore;
double spreads[];
int spread_idx = 0;

//+------------------------------------------------------------------+
//| Log event to file                                               |
//+------------------------------------------------------------------+
void LogEvent(string category, string message, bool forceFileLog = false)
  {
   bool shouldLogToFile = EnableLogging && (forceFileLog || category == "INIT" || category == "ERROR" || category == "SIM" || category == "TRADE");
   if(shouldLogToFile)
     {
      string filename = SymbolA + "-" + SymbolB + "-rehoboam-v4.log";
      int handle = FileOpen(filename, FILE_READ | FILE_WRITE | FILE_TXT | FILE_COMMON);
      if(handle != INVALID_HANDLE)
        {
         FileSeek(handle, 0, SEEK_END);
         FileWriteString(handle, "[" + TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + "] [" + category + "] " + message + "\n");
         FileFlush(handle);
         FileClose(handle);
        }
      else
        {
         Print("Failed to write to ", filename, ": ", GetLastError());
        }
     }
   Print("[" + category + "] " + message);
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
      if(!marketOpenA) LogEvent("ERROR", "Market closed for " + SymbolA);
      if(!marketOpenB) LogEvent("ERROR", "Market closed for " + SymbolB);
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
      LogEvent("ERROR", "Arrays have different sizes or are empty in MathCovariance.");
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
      LogEvent("ERROR", "Arrays have different sizes or are empty in MathCorrelation.");
      return 0.0;
     }
   int n = ArraySize(array1);
   double cov = MathCovariance(array1, array2);
   double std1 = MathStandardDeviation(array1);
   double std2 = MathStandardDeviation(array2);
   if(std1 == 0 || std2 == 0)
     {
      LogEvent("ERROR", "Standard deviation is zero in MathCorrelation.");
      return 0.0;
     }
   return cov / (std1 * std2);
  }

//+------------------------------------------------------------------+
//| Fetch and process spread data                                   |
//+------------------------------------------------------------------+
bool GetSpreadData(double &zscores[], double &mu, double &sigma)
  {
   double closesA[], closesB[];
   ArraySetAsSeries(closesA, true);
   ArraySetAsSeries(closesB, true);
   int m1_bars = RegressionPeriod;
   if(CopyClose(SymbolA, PERIOD_M1, 0, m1_bars, closesA) < m1_bars ||
      CopyClose(SymbolB, PERIOD_M1, 0, m1_bars, closesB) < m1_bars)
     {
      LogEvent("ERROR", "Insufficient M1 data for " + SymbolA + " or " + SymbolB);
      ArrayFree(closesA);
      ArrayFree(closesB);
      return false;
     }
   double spread_tmp[];
   ArrayResize(spread_tmp, m1_bars);
   for(int i = 0; i < m1_bars; i++)
     {
      spread_tmp[i] = closesA[i] - Beta * closesB[i];
     }
   // Extract most recent LookbackPeriod elements
   double spread_subset[];
   if(LookbackPeriod > m1_bars)
     {
      LogEvent("ERROR", "LookbackPeriod exceeds available M1 bars.");
      ArrayFree(spread_tmp);
      ArrayFree(closesA);
      ArrayFree(closesB);
      return false;
     }
   ArrayResize(spread_subset, LookbackPeriod);
   ArrayCopy(spread_subset, spread_tmp, 0, 0, LookbackPeriod);
   mu = MathMean(spread_subset);
   sigma = MathStandardDeviation(spread_subset);
   if(sigma == 0)
     {
      LogEvent("ERROR", "Zero spread standard deviation.");
      ArrayFree(spread_tmp);
      ArrayFree(closesA);
      ArrayFree(closesB);
      ArrayFree(spread_subset);
      return false;
     }
   ArrayResize(zscores, LookbackPeriod);
   for(int i = 0; i < LookbackPeriod; i++)
     {
      zscores[i] = (spread_subset[i] - mu) / sigma;
     }
   ArrayFree(spread_tmp);
   ArrayFree(closesA);
   ArrayFree(closesB);
   ArrayFree(spread_subset);
   LogEvent("SIM", "M1 data processed: " + IntegerToString(m1_bars) + " bars", true);
   return true;
  }

//+------------------------------------------------------------------+
//| Simulate Z-score paths                                           |
//+------------------------------------------------------------------+
void SimulateZScorePaths(double &zscores[], double &paths[])
  {
   ArrayResize(paths, SimNPaths * SimNSteps);
   MathSrand((int)TimeCurrent());
   for(int p = 0; p < SimNPaths; p++)
     {
      for(int t = 0; t < SimNSteps; t++)
        {
         int idx = MathRand() % ArraySize(zscores);
         paths[p * SimNSteps + t] = MathMin(MathMax(zscores[idx], -ZScoreMax), ZScoreMax);
        }
     }
   double min_z = paths[0], max_z = paths[0];
   for(int i = 0; i < SimNPaths * SimNSteps; i++)
     {
      min_z = MathMin(min_z, paths[i]);
      max_z = MathMax(max_z, paths[i]);
     }
   LogEvent("SIM", "Simulated Z-Score Range: Min=" + StringFormat("%.2f", min_z) + ", Max=" + StringFormat("%.2f", max_z), true);
  }

//+------------------------------------------------------------------+
//| Simulate portfolio                                              |
//+------------------------------------------------------------------+
void SimulatePortfolio(double &paths[], double mu, double sigma,
                      double &entry_zscores[], double &sharpe[], double &mean_pnl[],
                      double &win_rate[], double &max_drawdown[], int &trades[])
  {
   int n_zscores = ArraySize(entry_zscores);
   ArrayResize(sharpe, n_zscores);
   ArrayResize(mean_pnl, n_zscores);
   ArrayResize(win_rate, n_zscores);
   ArrayResize(max_drawdown, n_zscores);
   ArrayResize(trades, n_zscores);
   for(int z_idx = 0; z_idx < n_zscores; z_idx++)
     {
      double entry_z = entry_zscores[z_idx];
      double pnls[];
      ArrayResize(pnls, SimNPaths);
      int total_trades = 0;
      int wins = 0;
      double max_dd = 0.0;
      for(int p = 0; p < SimNPaths; p++)
        {
         double path_pnl = 0.0;
         int path_trades = 0;
         int position = 0;
         double initial_spread = 0.0;
         double entry_z_actual = 0.0;
         double equity = 0.0;
         double peak_equity = 0.0;
         for(int t = 1; t < SimNSteps; t++)
           {
            double z = paths[p * SimNSteps + t];
            if(position == 0)
              {
               if(z <= -entry_z)
                 {
                  position = 1;
                  initial_spread = mu + z * sigma;
                  entry_z_actual = z;
                  path_trades++;
                 }
               else if(z >= entry_z)
                 {
                  position = -1;
                  initial_spread = mu + z * sigma;
                  entry_z_actual = z;
                  path_trades++;
                 }
              }
            else
              {
               double spread = mu + z * sigma;
               double sl_z = (position == 1) ? entry_z_actual - (StopZScore - entry_z) : entry_z_actual + (StopZScore - entry_z);
               double tp_z = (position == 1) ? entry_z_actual + RiskRewardRatio * (StopZScore - entry_z) :
                             entry_z_actual - RiskRewardRatio * (StopZScore - entry_z);
               if((position == 1 && (z <= sl_z || z >= tp_z)) || (position == -1 && (z >= sl_z || z <= tp_z)))
                 {
                  double profit = (position == 1 ? spread - initial_spread : initial_spread - spread) / PipSize * PipValue * LotSize;
                  path_pnl += profit;
                  if(profit > 0) wins++;
                  equity += profit;
                  peak_equity = MathMax(peak_equity, equity);
                  max_dd = MathMax(max_dd, peak_equity - equity);
                  position = 0;
                  total_trades++;
                 }
              }
           }
         if(position != 0)
           {
            double spread = mu + paths[p * SimNSteps + SimNSteps - 1] * sigma;
            double profit = (position == 1 ? spread - initial_spread : initial_spread - spread) / PipSize * PipValue * LotSize;
            path_pnl += profit;
            if(profit > 0) wins++;
            equity += profit;
            peak_equity = MathMax(peak_equity, equity);
            max_dd = MathMax(max_dd, peak_equity - equity);
            total_trades++;
           }
         pnls[p] = path_pnl;
        }
      double mean = MathMean(pnls);
      double std = MathStandardDeviation(pnls);
      sharpe[z_idx] = (std == 0) ? 0.0 : mean / std;
      mean_pnl[z_idx] = mean;
      win_rate[z_idx] = (total_trades == 0) ? 0.0 : (double)wins / total_trades;
      max_drawdown[z_idx] = max_dd;
      trades[z_idx] = total_trades;
      LogEvent("SIM", "Entry Z-Score " + StringFormat("%.1f", entry_z) +
               ": " + IntegerToString(total_trades) + " trades, Avg Trades/Path: " + StringFormat("%.2f", (double)total_trades / SimNPaths) +
               ", Sharpe: " + StringFormat("%.4f", sharpe[z_idx]) +
               ", Mean PNL: " + StringFormat("%.2f", mean) +
               ", Win Rate: " + StringFormat("%.2f", win_rate[z_idx]) +
               ", Max Drawdown: " + StringFormat("%.2f", max_dd));
     }
  }

//+------------------------------------------------------------------+
//| Simulate optimal Z-score                                        |
//+------------------------------------------------------------------+
void SimulateOptimalZScore()
  {
   LogEvent("SIM", "Starting Z-score simulation", true);
   double zscores[];
   double mu, sigma;
   if(!GetSpreadData(zscores, mu, sigma))
     {
      LogEvent("ERROR", "Failed to get spread data for simulation. Using DefaultEntryZScore: " + StringFormat("%.1f", DefaultEntryZScore));
      EntryZScore = DefaultEntryZScore;
      return;
     }
   double paths[];
   SimulateZScorePaths(zscores, paths);
   double entry_zscores[];
   int n_zscores = (int)(ZScoreMax / ZScoreStep) + 1;
   ArrayResize(entry_zscores, n_zscores);
   for(int i = 0; i < n_zscores; i++)
     {
      entry_zscores[i] = i * ZScoreStep;
     }
   double sharpe[], mean_pnl[], win_rate[], max_drawdown[];
   int trades[];
   SimulatePortfolio(paths, mu, sigma, entry_zscores, sharpe, mean_pnl, win_rate, max_drawdown, trades);
   double max_sharpe = -DBL_MAX;
   double optimal_z = DefaultEntryZScore;
   int optimal_idx = 0;
   for(int i = 0; i < n_zscores; i++)
     {
      if(mean_pnl[i] > 0 && sharpe[i] > max_sharpe)
        {
         max_sharpe = sharpe[i];
         optimal_z = entry_zscores[i];
         optimal_idx = i;
        }
     }
   if(max_sharpe == -DBL_MAX)
     {
      double max_pnl = -DBL_MAX;
      for(int i = 0; i < n_zscores; i++)
        {
         if(mean_pnl[i] > max_pnl)
           {
            max_pnl = mean_pnl[i];
            optimal_z = entry_zscores[i];
            optimal_idx = i;
           }
        }
     }
   EntryZScore = MathMin(optimal_z, 8.0); // Cap at 8.0 as in v3
   // Write results to file
   string filename = SymbolA + "-" + SymbolB + "-rehoboam-v4-results.txt";
   int handle = FileOpen(filename, FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(handle != INVALID_HANDLE)
     {
      FileWrite(handle, "Simulation at ", TimeToString(TimeCurrent()),
                "\nOptimal Z-Score: ", StringFormat("%.1f", EntryZScore),
                "\nSharpe: ", StringFormat("%.4f", sharpe[optimal_idx]),
                "\nMean PNL: ", StringFormat("%.2f", mean_pnl[optimal_idx]),
                "\nWin Rate: ", StringFormat("%.2f", win_rate[optimal_idx]),
                "\nMax Drawdown: ", StringFormat("%.2f", max_drawdown[optimal_idx]),
                "\nTrades: ", trades[optimal_idx]);
      FileClose(handle);
      LogEvent("SIM", "Wrote results to " + filename, true);
     }
   else
     {
      LogEvent("ERROR", "Failed to write results to " + filename + ": " + IntegerToString(GetLastError()));
     }
   LogEvent("SIM", "Optimal Entry Z-Score: " + StringFormat("%.1f", EntryZScore) +
            ", Sharpe: " + StringFormat("%.4f", sharpe[optimal_idx]) +
            ", Mean PNL: " + StringFormat("%.2f", mean_pnl[optimal_idx]) +
            ", Win Rate: " + StringFormat("%.2f", win_rate[optimal_idx]) +
            ", Max Drawdown: " + StringFormat("%.2f", max_drawdown[optimal_idx]), true);
   ArrayFree(zscores);
   ArrayFree(paths);
   ArrayFree(entry_zscores);
   ArrayFree(sharpe);
   ArrayFree(mean_pnl);
   ArrayFree(win_rate);
   ArrayFree(max_drawdown);
   ArrayFree(trades);
  }

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Trade.SetExpertMagicNumber(MagicNumber);
   if(StringLen(SymbolA) == 0 || StringLen(SymbolB) == 0)
     {
      LogEvent("ERROR", "SymbolA and SymbolB must be specified.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(!SymbolSelect(SymbolA, true) || !SymbolSelect(SymbolB, true))
     {
      LogEvent("ERROR", "Symbols (" + SymbolA + ", " + SymbolB + ") not in Market Watch.");
      return(INIT_FAILED);
     }
   double testPriceA, testPriceB;
   if(!SymbolInfoDouble(SymbolA, SYMBOL_BID, testPriceA) || !SymbolInfoDouble(SymbolB, SYMBOL_BID, testPriceB))
     {
      LogEvent("ERROR", "Cannot retrieve price data for " + SymbolA + " or " + SymbolB + ".");
      return(INIT_FAILED);
     }
   if(!CalculateHedgeRatio())
     {
      LogEvent("ERROR", "Failed to calculate hedge ratio.");
      return(INIT_FAILED);
     }
   LogEvent("INIT", "Hedge Ratio (Beta): " + StringFormat("%.4f", Beta));
   LogEvent("INIT", "Parameters: LookbackPeriod=" + IntegerToString(LookbackPeriod) +
            ", RegressionPeriod=" + IntegerToString(RegressionPeriod) +
            ", DefaultEntryZScore=" + StringFormat("%.1f", DefaultEntryZScore) +
            ", MinEntryZScore=" + StringFormat("%.1f", MinEntryZScore) +
            ", StopZScore=" + StringFormat("%.1f", StopZScore) +
            ", RiskPercent=" + StringFormat("%.2f", RiskPercent) +
            ", MinCorrelation=" + StringFormat("%.2f", MinCorrelation) +
            ", RiskRewardRatio=" + StringFormat("%.2f", RiskRewardRatio) +
            ", SimNPaths=" + IntegerToString(SimNPaths) +
            ", SimNSteps=" + IntegerToString(SimNSteps) +
            ", SimIntervalMinutes=" + StringFormat("%.2f", SimIntervalMinutes) +
            ", EnableLogging=" + (EnableLogging ? "true" : "false") +
            ", DynamicEntryZScore=" + (DynamicEntryZScore ? "true" : "false"));
   ArrayResize(spreads, LookbackPeriod);
   ArrayInitialize(spreads, 0);
   if(DynamicEntryZScore)
     {
      EventSetTimer((int)(SimIntervalMinutes * 60));
      SimulateOptimalZScore();
     }
   else
     {
      EntryZScore = DefaultEntryZScore;
      LogEvent("INIT", "DynamicEntryZScore=false, using DefaultEntryZScore: " + StringFormat("%.1f", DefaultEntryZScore));
     }

   // Pre-load historical spreads for immediate runtime use
   double closesA[], closesB[];
   ArraySetAsSeries(closesA, true);
   ArraySetAsSeries(closesB, true);
   if(CopyClose(SymbolA, PERIOD_M1, 0, LookbackPeriod, closesA) >= LookbackPeriod &&
      CopyClose(SymbolB, PERIOD_M1, 0, LookbackPeriod, closesB) >= LookbackPeriod)
     {
      for(int i = 0; i < LookbackPeriod; i++)
        {
         spreads[i] = closesA[i] - Beta * closesB[i];
        }
      spread_idx = LookbackPeriod;
      LogEvent("INIT", "Pre-loaded " + IntegerToString(LookbackPeriod) + " historical spreads for runtime Z-scores.");
     }
   else
     {
      LogEvent("ERROR", "Failed to pre-load historical spreads; will build prospectively.");
     }
   ArrayFree(closesA);
   ArrayFree(closesB);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert timer function                                            |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(DynamicEntryZScore)
     {
      SimulateOptimalZScore();
     }
  }

//+------------------------------------------------------------------+
//| Count open spreads (based on SymbolA positions)                  |
//+------------------------------------------------------------------+
int CountOpenSpreads()
  {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i))
        {
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetString(POSITION_SYMBOL) == SymbolA)
           {
            count++;
           }
        }
     }
   return count;
  }

//+------------------------------------------------------------------+
//| Count open SymbolB positions                                     |
//+------------------------------------------------------------------+
int CountSymbolBPositions()
  {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i))
        {
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetString(POSITION_SYMBOL) == SymbolB)
           {
            count++;
           }
        }
     }
   return count;
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsMarketOpen()) return;

   // Get current tick data
   double bidA = SymbolInfoDouble(SymbolA, SYMBOL_BID);
   double askA = SymbolInfoDouble(SymbolA, SYMBOL_ASK);
   double bidB = SymbolInfoDouble(SymbolB, SYMBOL_BID);
   double askB = SymbolInfoDouble(SymbolB, SYMBOL_ASK);
   datetime current_time = TimeCurrent();

   // Check if new M1 bar
   static datetime last_bar = 0;
   datetime current_bar = current_time - (current_time % 60);
   if(current_bar > last_bar)
     {
      last_bar = current_bar;

      // Update spread array
      double current_spread = (askA + bidA) / 2.0 - Beta * (askB + bidB) / 2.0;
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

      // Calculate and log Z-score if enough data
      if(spread_idx >= LookbackPeriod)
        {
         double mu = MathMean(spreads);
         double sigma = MathStandardDeviation(spreads);
         if(sigma == 0)
           {
            LogEvent("ERROR", "Spread standard deviation is zero.");
            return;
           }
         double zScore = (current_spread - mu) / sigma;
         LogEvent("TRADE", "Current Z-Score (M1): " + StringFormat("%.4f", zScore), true);

         // Check for mismatched positions (partial opens)
         int currentSpreads = CountOpenSpreads();
         int countB = CountSymbolBPositions();
         if(currentSpreads != countB)
           {
            LogEvent("ERROR", "Mismatched pair positions detected (A: " + IntegerToString(currentSpreads) + ", B: " + IntegerToString(countB) + "). Closing all positions.", true);
            CloseAllPositions();
            return;
           }

         int direction = GetPositionDirection();
         if(direction != 0)
           {
            bool shouldClose = false;
            double currentProfit = CalculatePairProfit();
            if(direction == 1)
              {
               if(SL_Type == SL_ZScore)
                 {
                  double sl_z = EntryZ - (StopZScore - EntryZScore);
                  double tp_z = (TP_Type == TP_Multiple) ? EntryZ + RiskRewardRatio * (StopZScore - EntryZScore) : TakeProfitZScore;
                  if(zScore <= sl_z || zScore >= tp_z)
                    {
                     shouldClose = true;
                     LogEvent("TRADE", "Closing long spread: Z-Score=" + StringFormat("%.4f", zScore) +
                              ", Profit=" + StringFormat("%.2f", currentProfit) +
                              ", SL_Z=" + StringFormat("%.4f", sl_z) +
                              ", TP_Z=" + StringFormat("%.4f", tp_z), true);
                    }
                 }
               else
                 {
                  double sl_amount = -(StopLossPercent / 100.0) * EntryEquity;
                  double tp_amount = -sl_amount * RiskRewardRatio;
                  if(currentProfit <= sl_amount || currentProfit >= tp_amount)
                    {
                     shouldClose = true;
                     LogEvent("TRADE", "Closing long spread: Profit=" + StringFormat("%.2f", currentProfit) +
                              ", SL_Amount=" + StringFormat("%.2f", sl_amount) +
                              ", TP_Amount=" + StringFormat("%.2f", tp_amount), true);
                    }
                 }
              }
            else if(direction == -1)
              {
               if(SL_Type == SL_ZScore)
                 {
                  double sl_z = EntryZ + (StopZScore - EntryZScore);
                  double tp_z = (TP_Type == TP_Multiple) ? EntryZ - RiskRewardRatio * (StopZScore - EntryZScore) : TakeProfitZScore;
                  if(zScore >= sl_z || zScore <= tp_z)
                    {
                     shouldClose = true;
                     LogEvent("TRADE", "Closing short spread: Z-Score=" + StringFormat("%.4f", zScore) +
                              ", Profit=" + StringFormat("%.2f", currentProfit) +
                              ", SL_Z=" + StringFormat("%.4f", sl_z) +
                              ", TP_Z=" + StringFormat("%.4f", tp_z), true);
                    }
                 }
               else
                 {
                  double sl_amount = -(StopLossPercent / 100.0) * EntryEquity;
                  double tp_amount = -sl_amount * RiskRewardRatio;
                  if(currentProfit <= sl_amount || currentProfit >= tp_amount)
                    {
                     shouldClose = true;
                     LogEvent("TRADE", "Closing short spread: Profit=" + StringFormat("%.2f", currentProfit) +
                              ", SL_Amount=" + StringFormat("%.2f", sl_amount) +
                              ", TP_Amount=" + StringFormat("%.2f", tp_amount), true);
                    }
                 }
              }
            if(shouldClose)
              {
               CloseAllPositions();
              }
           }

         // Check for opening new position
         if(MathAbs(zScore) >= MinEntryZScore && MathAbs(zScore) >= EntryZScore)
           {
            currentSpreads = CountOpenSpreads();
            if(currentSpreads < MaxOpenPositions)
              {
               int proposed_direction = 0;
               if(zScore <= -EntryZScore) proposed_direction = 1;
               else if(zScore >= EntryZScore) proposed_direction = -1;
               if(direction == 0 || direction == proposed_direction)
                 {
                  double lotsA = CalculateLots(sigma);
                  double lotsB = MathAbs(Beta) * lotsA;
                  LogEvent("TRADE", "Z-Score (M1): " + StringFormat("%.4f", zScore) + ", Opening " + (proposed_direction == 1 ? "long" : "short") + " spread.", true);
                  if(proposed_direction == 1)
                    {
                     OpenLongSpread(lotsA, lotsB, zScore);
                    }
                  else
                    {
                     OpenShortSpread(lotsA, lotsB, zScore);
                    }
                 }
               else
                 {
                  LogEvent("TRADE", "Cannot open new position in opposite direction while positions open.");
                 }
              }
            else
              {
               LogEvent("TRADE", "Maximum open positions reached (" + IntegerToString(currentSpreads) + "/" + IntegerToString(MaxOpenPositions) + "). No new trade opened.");
              }
           }
         else
           {
            if(MathAbs(zScore) < MinEntryZScore)
              {
               LogEvent("TRADE", "Z-Score (M1): " + StringFormat("%.4f", zScore) +
                        " below MinEntryZScore (" + StringFormat("%.1f", MinEntryZScore) + "). No trade opened.", true);
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
   if(CopyClose(SymbolA, PERIOD_M1, 0, RegressionPeriod, closesA) < RegressionPeriod ||
      CopyClose(SymbolB, PERIOD_M1, 0, RegressionPeriod, closesB) < RegressionPeriod)
     {
      LogEvent("ERROR", "Insufficient historical data for " + SymbolA + " or " + SymbolB);
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
   LogEvent("INIT", "Pair Correlation: " + StringFormat("%.4f", correlation));
   if(correlation < MinCorrelation && !BypassCorrelationCheck)
     {
      LogEvent("ERROR", "Correlation (" + StringFormat("%.4f", correlation) + ") below threshold (" + StringFormat("%.2f", MinCorrelation) + ").");
      return false;
     }
   double cov = MathCovariance(closesA, closesB);
   double varB = MathVariance(closesB);
   if(varB == 0)
     {
      LogEvent("ERROR", "Variance of " + SymbolB + " is zero.");
      return false;
     }
   Beta = cov / varB;
   LogEvent("INIT", "Calculated Beta: " + StringFormat("%.4f", Beta));
   return true;
  }

//+------------------------------------------------------------------+
//| Calculate lots for Asset A based on risk                         |
//+------------------------------------------------------------------+
double CalculateLots(double sigma)
  {
   if(sigma <= 0)
     {
      LogEvent("ERROR", "Sigma is zero or negative.");
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
      LogEvent("ERROR", "Invalid tick size or value for " + SymbolA);
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
      LogEvent("ERROR", "Invalid lot sizes for long spread (A: " + StringFormat("%.2f", lotsA) + ", B: " + StringFormat("%.2f", lotsB) + ")");
      return;
     }
   double lotStepB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_STEP);
   if(lotStepB > 0)
      lotsB = MathFloor(lotsB / lotStepB) * lotStepB;
   double minLotB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MIN);
   double maxLotB = MathMin(SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MAX), MaxLots);
   lotsB = MathMax(minLotB, MathMin(maxLotB, lotsB));
   // Adjust SymbolB trade direction based on Beta sign
   ENUM_ORDER_TYPE orderTypeB = (Beta >= 0) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   if(Trade.PositionOpen(SymbolA, ORDER_TYPE_BUY, lotsA, 0, 0, 0, "Pairs Long Spread"))
     {
      LogEvent("TRADE", "Opened BUY position for " + SymbolA + ", Lots=" + StringFormat("%.2f", lotsA) + ", Z-Score=" + StringFormat("%.4f", zScore), true);
     }
   else
     {
      LogEvent("ERROR", "Error opening BUY position for " + SymbolA + ": " + IntegerToString(GetLastError()));
     }
   if(Trade.PositionOpen(SymbolB, orderTypeB, lotsB, 0, 0, 0, "Pairs Long Spread"))
     {
      LogEvent("TRADE", "Opened " + (orderTypeB == ORDER_TYPE_SELL ? "SELL" : "BUY") + " position for " + SymbolB + ", Lots=" + StringFormat("%.2f", lotsB) + ", Z-Score=" + StringFormat("%.4f", zScore), true);
     }
   else
     {
      LogEvent("ERROR", "Error opening " + (orderTypeB == ORDER_TYPE_SELL ? "SELL" : "BUY") + " position for " + SymbolB + ": " + IntegerToString(GetLastError()));
     }
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
      LogEvent("ERROR", "Invalid lot sizes for short spread (A: " + StringFormat("%.2f", lotsA) + ", B: " + StringFormat("%.2f", lotsB) + ")");
      return;
     }
   double lotStepB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_STEP);
   if(lotStepB > 0)
      lotsB = MathFloor(lotsB / lotStepB) * lotStepB;
   double minLotB = SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MIN);
   double maxLotB = MathMin(SymbolInfoDouble(SymbolB, SYMBOL_VOLUME_MAX), MaxLots);
   lotsB = MathMax(minLotB, MathMin(maxLotB, lotsB));
   // Adjust SymbolB trade direction based on Beta sign
   ENUM_ORDER_TYPE orderTypeB = (Beta >= 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(Trade.PositionOpen(SymbolA, ORDER_TYPE_SELL, lotsA, 0, 0, 0, "Pairs Short Spread"))
     {
      LogEvent("TRADE", "Opened SELL position for " + SymbolA + ", Lots=" + StringFormat("%.2f", lotsA) + ", Z-Score=" + StringFormat("%.4f", zScore), true);
     }
   else
     {
      LogEvent("ERROR", "Error opening SELL position for " + SymbolA + ": " + IntegerToString(GetLastError()));
     }
   if(Trade.PositionOpen(SymbolB, orderTypeB, lotsB, 0, 0, 0, "Pairs Short Spread"))
     {
      LogEvent("TRADE", "Opened " + (orderTypeB == ORDER_TYPE_BUY ? "BUY" : "SELL") + " position for " + SymbolB + ", Lots=" + StringFormat("%.2f", lotsB) + ", Z-Score=" + StringFormat("%.4f", zScore), true);
     }
   else
     {
      LogEvent("ERROR", "Error opening " + (orderTypeB == ORDER_TYPE_BUY ? "BUY" : "SELL") + " position for " + SymbolB + ": " + IntegerToString(GetLastError()));
     }
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
            if(Trade.PositionClose(PositionGetInteger(POSITION_TICKET)))
              {
               LogEvent("TRADE", "Closed position for " + PositionGetString(POSITION_SYMBOL) + ", Profit=" + StringFormat("%.2f", PositionGetDouble(POSITION_PROFIT)), true);
              }
            else
              {
               LogEvent("ERROR", "Error closing position for " + PositionGetString(POSITION_SYMBOL) + ": " + IntegerToString(GetLastError()));
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+