#!/usr/bin/env python3
"""
CSV to Parquet Converter

This script processes CSV files and converts them to parquet format.
If dask=0: saves as a single parquet file
If dask=1: splits into partitions of 1000 sequences each

Also creates a targets CSV file with Accession ID and Target columns.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd


def read_csv_file(csv_path):
    """
    Read CSV file and validate required columns.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing only the required columns
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check for required columns
        required_columns = ['Accession ID', 'Sequence', 'Target']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        
        # Keep only the required columns
        df = df[required_columns]
        
        return df
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)


def save_single_parquet(df, output_path):
    """
    Save DataFrame to a single parquet file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        output_path (str): Path where to save the parquet file
    """
    # Save only Accession ID and Sequence columns to parquet
    parquet_df = df[['Accession ID', 'Sequence']]
    parquet_df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Saved {len(df)} sequences to {output_path}")


def save_partitioned_parquet(df, output_dir, partition_size=1000):
    """
    Save DataFrame to multiple parquet files, each containing partition_size rows.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        output_dir (str): Directory where to save the parquet files
        partition_size (int): Number of rows per partition
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split DataFrame into partitions
    total_rows = len(df)
    for i in range(0, total_rows, partition_size):
        partition_num = (i // partition_size) + 1
        partition_df = df.iloc[i:i + partition_size]
        
        # Save only Accession ID and Sequence columns to parquet
        parquet_df = partition_df[['Accession ID', 'Sequence']]
        output_path = os.path.join(output_dir, f'partition_{partition_num}.parquet')
        parquet_df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Saved partition {partition_num} with {len(partition_df)} sequences to {output_path}")


def save_targets_csv(df, output_path):
    """
    Save Accession ID and Target columns to a separate CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        output_path (str): Path where to save the targets CSV file
    """
    targets_df = df[['Accession ID', 'Target']].copy()
    targets_df.to_csv(output_path, index=False)
    print(f"Saved {len(targets_df)} targets to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV file to parquet format')
    parser.add_argument('path', help='Path to the input CSV file')
    parser.add_argument('dask', type=int, choices=[0, 1], 
                       help='0: save as single parquet file, 1: save as partitioned parquet files')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.path):
        print(f"Error: File {args.path} does not exist")
        sys.exit(1)
    
    # Get the base name of the CSV file (without extension)
    csv_name = Path(args.path).stem
    
    # Create output directory path
    output_base_dir = Path('../data')
    output_base_dir.mkdir(exist_ok=True)
    
    # Read CSV file
    print(f"Reading CSV file: {args.path}")
    df = read_csv_file(args.path)
    print(f"Found {len(df)} sequences with columns: {list(df.columns)}")
    
    # Always create targets CSV file
    targets_output_path = output_base_dir / f"{csv_name}_targets.csv"
    save_targets_csv(df, targets_output_path)
    
    if args.dask == 0:
        # Save as single parquet file
        output_path = output_base_dir / f"{csv_name}.parquet"
        save_single_parquet(df, output_path)
        
    elif args.dask == 1:
        # Save as partitioned parquet files
        output_dir = output_base_dir / csv_name
        save_partitioned_parquet(df, output_dir)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
