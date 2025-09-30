import pandas as pd
import os


def parse_mt5_csv(csv_file):
    """Parse MT5 optimization CSV file and return a DataFrame."""
    df = pd.read_csv(csv_file)
    return df


def analyze_optimization_folder(folder_path):
    """Analyze all MT5 CSV optimization files in a folder and return aggregated DataFrame."""
    all_passes = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            period = filename.split('.')[0]  # Assume filename like 'Feb2024.csv'
            df = parse_mt5_csv(file_path)
            df['Period'] = period
            all_passes.append(df)

    if not all_passes:
        print("Error: No CSV files found in the specified folder.")
        return pd.DataFrame()

    df = pd.concat(all_passes, ignore_index=True)

    # Print available columns for debugging
    print("Available columns in DataFrame:", df.columns.tolist())

    # Normalize column names (handle case sensitivity and spaces)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('%', '')

    # Explicitly rename 'equity_dd_' to 'equity_dd_percent'
    if 'equity_dd_' in df.columns:
        df = df.rename(columns={'equity_dd_': 'equity_dd_percent'})

    # Check for required columns
    required_cols = ['entryzscore', 'stoplosspercent', 'profit', 'equity_dd_percent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns {missing_cols}. Returning raw DataFrame.")
        return df

    # Convert relevant columns to numeric
    numeric_cols = ['result', 'profit', 'expected_payoff', 'profit_factor',
                    'recovery_factor', 'sharpe_ratio', 'equity_dd_percent',
                    'trades', 'entryzscore', 'stoplosspercent', 'forward_result', 'back_result']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in key columns
    df = df.dropna(subset=['entryzscore', 'stoplosspercent', 'profit', 'equity_dd_percent'])

    # Group by parameters and compute aggregates
    grouped = df.groupby(['entryzscore', 'stoplosspercent']).agg({
        'profit': ['mean', 'median', 'min', 'max', 'count'],
        'equity_dd_percent': ['mean', 'median', 'max'],
        'sharpe_ratio': ['mean', 'median'],
        'recovery_factor': ['mean', 'median'],
        'trades': ['mean'],
        'forward_result': ['mean', 'median'],
        'back_result': ['mean', 'median']
    }).reset_index()

    # Flatten multi-index columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    # Filter for reasonable trade counts (20–200 trades for 2-month period, adjusted for sample data)
    grouped = grouped[(grouped['trades_mean'] >= 20) & (grouped['trades_mean'] <= 200)]

    # Sort by high mean profit, low mean DD, high Sharpe, and high forward result
    grouped = grouped.sort_values(
        by=['profit_mean', 'equity_dd_percent_mean', 'sharpe_ratio_mean', 'forward_result_mean'],
        ascending=[False, True, False, False]
    )

    return grouped

if __name__ == "__main__":
    folder_path = 'optimization files'  # Folder with CSV files
    results = analyze_optimization_folder(folder_path)
    print(results)
    results.to_csv('aggregated_optimization_results.csv', index=False)
