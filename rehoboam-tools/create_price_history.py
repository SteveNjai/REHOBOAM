
import pandas as pd
import os


def create_price_history(folder_path, output_file='price_history.csv'):
    """Create price_history.csv by merging Close prices from all CSV files in the folder."""
    dataframes = {}

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            symbol = filename.split('.csv')[0].upper()  # Extract symbol from filename, e.g., 'xauusd' -> 'XAUUSD'
            file_path = os.path.join(folder_path, filename)

            # Read the tab-delimited CSV
            df = pd.read_csv(file_path, sep='\t')

            # Rename columns to remove <> if present
            df.columns = [col.strip('<>') for col in df.columns]

            # Combine DATE and TIME into a single datetime column
            df['Date'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')

            # Extract Close column and rename to symbol
            df = df[['Date', 'CLOSE']].rename(columns={'CLOSE': symbol}).set_index('Date')

            dataframes[symbol] = df

    # Merge all dataframes on Date index
    merged_df = pd.concat(dataframes.values(), axis=1)

    # Forward fill any missing values
    merged_df = merged_df.fillna(method='ffill')

    # Export to price_history.csv
    merged_df.to_csv(output_file)

    print(f"price_history.csv exported successfully to {output_file}")


# Example usage
if __name__ == "__main__":
    folder_path = 'history'  # Your folder path
    create_price_history(folder_path)