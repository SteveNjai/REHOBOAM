import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import itertools
import warnings

warnings.filterwarnings("ignore")


def load_price_data(csv_file):
    """Load price history from CSV file with Date in MM/DD/YYYY HH:MM format."""
    df = pd.read_csv(csv_file, parse_dates=['Date'], date_format='%m/%d/%Y %H:%M')
    df = df.set_index('Date')
    # Forward fill missing values
    df = df.fillna(method='ffill')
    # Remove rows with NaNs or zeros
    df = df.dropna()
    df = df[(df != 0).all(axis=1)]
    # Normalization disabled to preserve original price scale
    # df = df / df.mean()
    return df


def engle_granger_test(priceA, priceB):
    """Perform Engle-Granger two-step cointegration test."""
    model = OLS(priceA, add_constant(priceB)).fit()
    beta = model.params[1]
    residuals = priceA - beta * priceB
    adf_result = adfuller(residuals)
    p_value = adf_result[1]
    return beta, p_value, residuals


def johansen_test(prices, det_order=0, k_ar_diff=1):
    """Perform Johansen cointegration test on multiple series."""
    try:
        result = coint_johansen(prices, det_order, k_ar_diff)
        return result.trace_stat, result.max_eig_stat, result.trace_stat_crit_vals, result.max_eig_stat_crit_vals
    except np.linalg.LinAlgError:
        print("Warning: Johansen test failed due to SVD non-convergence. Skipping Johansen results.")
        return None, None, None, None


def calculate_half_life(spread):
    """Calculate half-life of mean reversion for the spread."""
    lagged_spread = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()
    model = OLS(delta_spread, lagged_spread).fit()
    theta = -model.params[0]
    half_life = np.log(2) / theta
    return half_life


def hierarchical_clustering(returns_df):
    """Perform hierarchical clustering on asset returns."""
    corr_matrix = returns_df.corr()
    dist = pdist(1 - corr_matrix, 'euclidean')
    link = linkage(dist, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(link, labels=returns_df.columns)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Assets')
    plt.ylabel('Distance')
    plt.show()
    return link


def calculate_spread_volatility(spread):
    """Calculate volatility (std dev) of the spread."""
    return np.std(spread)


def screen_pairs(df, min_correlation=0.7):
    """Screen for cointegrated pairs and return data."""
    symbols = df.columns
    results = []

    for symA, symB in itertools.combinations(symbols, 2):
        priceA = df[symA].dropna()
        priceB = df[symB].dropna()
        min_len = min(len(priceA), len(priceB))
        priceA = priceA[-min_len:]
        priceB = priceB[-min_len:]

        # Correlation
        correlation = priceA.corr(priceB)
        if correlation < min_correlation:
            continue

        # Engle-Granger
        beta, p_value, residuals = engle_granger_test(priceA, priceB)
        if p_value < 0.05:
            # Half-life
            half_life = calculate_half_life(pd.Series(residuals))

            # Spread volatility
            spread = priceA - beta * priceB
            sigma = calculate_spread_volatility(spread)

            results.append({
                'Pair': f"{symA}-{symB}",
                'Correlation': correlation,
                'Beta': beta,
                'EG_PValue': p_value,
                'HalfLife': half_life,
                'SpreadVolatility': sigma
            })

    results_df = pd.DataFrame(results)

    # Johansen for all symbols (if more than 2)
    if len(symbols) > 2:
        prices = df.to_numpy()
        trace_stat, max_eig_stat, trace_cv, max_eig_cv = johansen_test(prices)
        if trace_stat is not None:
            results_df['Johansen_Trace'] = [trace_stat] * len(results_df)
            results_df['Johansen_MaxEig'] = [max_eig_stat] * len(results_df)

    # Clustering
    returns_df = df.pct_change().dropna()
    if not returns_df.empty:
        hierarchical_clustering(returns_df)

    return results_df


def main(csv_file, output_txt='cointegration_results.txt'):
    """Main function to load data, screen pairs, and save results to text file."""
    df = load_price_data(csv_file)
    screened_data = screen_pairs(df)
    # Save results to text file
    with open(output_txt, 'w') as f:
        f.write("Cointegration Screening Results\n")
        f.write("==========================\n\n")
        f.write(screened_data.to_string(index=False))
        f.write("\n\nNote: HalfLife is in hours. Johansen_Trace and Johansen_MaxEig may be omitted if the test failed.")
    print(f"Results saved to {output_txt}")
    return screened_data


if __name__ == "__main__":
    csv_file = 'price_history.csv'
    data = main(csv_file)
    print(data)
