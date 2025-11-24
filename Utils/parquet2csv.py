import pandas as pd
import os

def parquet_to_csv(parquet_file_path, csv_file_path):
    """
    Converts a Parquet file to a CSV file.

    Args:
        parquet_file_path (str): The path to the input Parquet file.
        csv_file_path (str): The path to the output CSV file.
    """
    try:
        df = pd.read_parquet(parquet_file_path)
        df.to_csv(csv_file_path, index=False)
        print(f"Successfully converted '{parquet_file_path}' to '{csv_file_path}'")
    except Exception as e:
        print(f"Error converting '{parquet_file_path}' to CSV: {e}")

if __name__ == "__main__":
    # Example usage:
    # Define input and output paths
    input_parquet_file = "Data/Raw/20240430.parquet"  # Replace with your .parquet file path
    output_csv_file = "sample_output.csv"        # Replace with your desired .csv file path

    # Create a dummy parquet file for testing if it doesn't exist
    if not os.path.exists(input_parquet_file):
        print(f"Creating a dummy parquet file at {input_parquet_file} for demonstration.")
        dummy_data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}
        dummy_df = pd.DataFrame(dummy_data)
        os.makedirs(os.path.dirname(input_parquet_file) or '.', exist_ok=True)
        dummy_df.to_parquet(input_parquet_file, index=False)

    parquet_to_csv(input_parquet_file, output_csv_file)
