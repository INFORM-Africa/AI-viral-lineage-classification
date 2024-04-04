import argparse
import pandas as pd
import numpy as np
import pyfftw
import random
from tqdm import tqdm
import os
from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides, load_sequences, save_features_to_parquet

def anti_symmetric_padding(seq, target_length):
    """
    Apply anti-symmetric padding to a sequence to reach a target length.

    Parameters:
    - seq (np.array): The original sequence as a NumPy array.
    - target_length (int): The desired length after padding.

    Returns:
    - np.array: The padded sequence with anti-symmetric elements.
    """
    if len(seq) >= target_length:
        return seq  # Return the original sequence if it's already long enough

    pad_size = target_length - len(seq)
    repeats = -(-pad_size // len(seq))  # Ceiling division to ensure enough repeats
    extended_seq = np.tile(seq[::-1], repeats) * (-1) ** np.arange(1, len(seq) * repeats + 1)
    pad_seq = extended_seq[:pad_size]

    return np.concatenate((seq, pad_seq))

def numeric_transform(sq, mapping):
    """
    Transform a nucleotide sequence into a numeric sequence based on a provided mapping.

    Parameters:
    - sequence (str): The nucleotide sequence (e.g., 'AGCT').
    - mapping (dict): Mapping from nucleotide characters to numeric values.

    Returns:
    - np.array: The numeric sequence as a NumPy array of floats.
    """
    
    # Create a NumPy array of the same length as the input sequence
    numSeq = np.zeros(len(sq), dtype=np.float32)

    # Map characters to indices: A->0, C->1, G->2, T->3, others->-1 (which will be ignored)
    char_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = np.array([char_to_index.get(char, -1) for char in sq])

    # Use boolean indexing to set values only for valid indices
    valid_indices = (indices >= 0)
    numSeq[valid_indices] = mapping[indices[valid_indices]]

    return numSeq

def fourier_transform(sequence):
    """
    Compute the Fourier transform of a numeric sequence and return its magnitude spectrum.

    Parameters:
    - sequence (np.array): Numeric sequence to transform.

    Returns:
    - np.array: Magnitude spectrum of the Fourier transform.
    """
    fft_object = pyfftw.builders.fft(sequence)
    fourier = fft_object()
    magnitude_spectrum = np.abs(fourier)

    return magnitude_spectrum

def process_sequence(sequence, max_length, mapping):
    """
    Process a nucleotide sequence: transform to numeric, apply padding if necessary, and compute the Fourier magnitude spectrum.

    Parameters:
    - sequence (str): The nucleotide sequence to process.
    - max_length (int): The maximum length for padding or truncating the sequence.
    - mapping (dict): Mapping from nucleotide characters to numeric values.

    Returns:
    - np.array: Fourier magnitude spectrum of the processed sequence.
    """
    numeric_sequence = numeric_transform(sequence, mapping)

    if len(numeric_sequence) < max_length:
        numeric_sequence = anti_symmetric_padding(numeric_sequence, max_length)
    else:
        numeric_sequence = numeric_sequence[:max_length]

    magnitude_spectrum = fourier_transform(numeric_sequence).astype(np.float16)

    return magnitude_spectrum

def main():
    parser = argparse.ArgumentParser(description='Process DNA sequences using Chaos Game Representation (CGR).')
    parser.add_argument('-d', '--Degenerate', choices=['replace', 'remove'], required=True,
                        help='Specify how to handle degenerate nucleotides: Replace or Remove.')
    parser.add_argument('-n', '--Numeric', type=str, choices=["Real", "PP", "JustA", "EEIP"], required=True,
                        help='Type of Numeric Mapping.')
    parser.add_argument('-v', '--Data', choices=['SARS-CoV-2', 'HIV'], required=True,
                        help='Specify the virus dataset.')
    args = parser.parse_args()
    
    print(f"Running DSP with {args.Degenerate} option and {args.Numeric} mapping.")

    sequence_df = load_sequences(args)
    
    lengths = sequence_df["Sequence"].str.len()
    median_length = int(lengths.median())

    if args.Numeric == "Real":
        mapping = np.array([-1.5, 0.5, -0.5, 1.5], dtype=np.float32)  # A, C, G, T mapping in order
    elif args.Numeric == "PP":
        mapping = np.array([-1, 1, -1, 1], dtype=np.float32)  # A, C, G, T mapping in order
    elif args.Numeric == "JustA":
        mapping = np.array([1, 0, 0, 0], dtype=np.float32)  # A, C, G, T mapping in order
    elif args.Numeric == "EIIP":
        mapping = np.array([0.1260, 0.1340, 0.0806, 0.1335], dtype=np.float32)  # A, C, G, T mapping in order

    sequence_df = sequence_df.reset_index()

    GSP_data = np.zeros((len(sequence_df), median_length), dtype=np.float16)
    for i in tqdm(range(len(sequence_df)), desc="Processing sequences"):
        GSP_data[i] = process_sequence(sequence_df["Sequence"][i], median_length, mapping)  # Assuming 'mapping' is defined elsewhere

    GSP_data = pd.DataFrame(GSP_data)
    
    # Create DataFrame from k-mer features and add target labels
    targets = sequence_df['Target'].to_list()
    train_labels = sequence_df['Train'].to_list()
    save_features_to_parquet(GSP_data, args, targets, train_labels)
    
if __name__ == "__main__":
    main()