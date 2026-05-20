#!/usr/bin/env python3
"""
FASTA to Parquet Converter

This script processes FASTA files and converts them to parquet format.
If dask=0: saves as a single parquet file
If dask=1: splits into partitions of 1000 sequences each
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from Bio import SeqIO


def parse_fasta_file(fasta_path):
    """
    Parse FASTA file and extract Accession IDs and Sequences.
    
    Args:
        fasta_path (str): Path to the FASTA file
        
    Returns:
        list: List of tuples containing (accession_id, sequence)
    """
    sequences = []
    
    try:
        with open(fasta_path, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Extract accession ID from the record ID
                accession_id = record.id
                sequence = str(record.seq)
                sequences.append((accession_id, sequence))
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        sys.exit(1)
    
    return sequences


def save_single_parquet(sequences, output_path):
    """
    Save all sequences to a single parquet file.
    
    Args:
        sequences (list): List of (accession_id, sequence) tuples
        output_path (str): Path where to save the parquet file
    """
    df = pd.DataFrame(sequences, columns=['Accession ID', 'Sequence'])
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Saved {len(sequences)} sequences to {output_path}")


def save_partitioned_parquet(sequences, output_dir, partition_size=1000):
    """
    Save sequences to multiple parquet files, each containing partition_size sequences.
    
    Args:
        sequences (list): List of (accession_id, sequence) tuples
        output_dir (str): Directory where to save the parquet files
        partition_size (int): Number of sequences per partition
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split sequences into partitions
    for i in range(0, len(sequences), partition_size):
        partition_num = (i // partition_size) + 1
        partition_sequences = sequences[i:i + partition_size]
        
        df = pd.DataFrame(partition_sequences, columns=['Accession ID', 'Sequence'])
        output_path = os.path.join(output_dir, f'partition_{partition_num}.parquet')
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Saved partition {partition_num} with {len(partition_sequences)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert FASTA file to parquet format')
    parser.add_argument('path', help='Path to the input FASTA file')
    parser.add_argument('dask', type=int, choices=[0, 1], 
                       help='0: save as single parquet file, 1: save as partitioned parquet files')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.path):
        print(f"Error: File {args.path} does not exist")
        sys.exit(1)
    
    # Get the base name of the FASTA file (without extension)
    fasta_name = Path(args.path).stem
    
    # Create output directory path
    output_base_dir = Path('../data')
    output_base_dir.mkdir(exist_ok=True)
    
    # Parse FASTA file
    print(f"Parsing FASTA file: {args.path}")
    sequences = parse_fasta_file(args.path)
    print(f"Found {len(sequences)} sequences")
    
    if args.dask == 0:
        # Save as single parquet file
        output_path = output_base_dir / f"{fasta_name}.parquet"
        save_single_parquet(sequences, output_path)
        
    elif args.dask == 1:
        # Save as partitioned parquet files
        output_dir = output_base_dir / fasta_name
        save_partitioned_parquet(sequences, output_dir)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
