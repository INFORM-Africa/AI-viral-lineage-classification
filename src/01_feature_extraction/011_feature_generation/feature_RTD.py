import argparse
import pandas as pd
import numpy as np
import copy
from itertools import product
from tqdm import tqdm
import os

from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides, load_sequences, save_features_to_parquet

def precompute_kmer_dict(k):
    """
    Precompute a dictionary with statistics for all possible k-mers.

    Parameters:
    - k (int): Length of the k-mers.

    Returns:
    - dict: A dictionary with keys as k-mers and values as dictionaries containing 'sum', 'count', and 'last_seen' initialized.
    """
    alphabet = 'ACGT'
    all_kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    kmer_stats = {kmer: {'sum': 0, 'count': 0, 'last_seen': -1} for kmer in all_kmers}

    return kmer_stats

def compute_rtd_feature_vector(genome, k, kmer_stats):
    """
    Compute the feature vector for a genome based on Relative Time Distance (RTD) of k-mers.

    Parameters:
    - genome (str): The genomic sequence to be analyzed.
    - k (int): The length of k-mers to consider.
    - kmer_stats (dict): A precomputed dictionary of k-mer statistics.

    Returns:
    - np.array: A feature vector representing the mean and standard deviation of RTD for each k-mer.
    """
    for i in range(len(genome) - k + 1):
        kmer = genome[i:i+k]
        if kmer in kmer_stats:
            stats = kmer_stats[kmer]
            if stats['last_seen'] != -1:
                distance = i - stats['last_seen']
                stats['sum'] += distance
                stats['count'] += 1
            stats['last_seen'] = i

    feature_vector = np.zeros(2 * len(kmer_stats), dtype=np.float32)
    for idx, (kmer, stats) in enumerate(kmer_stats.items()):
        if stats['count'] > 0:
            mean = stats['sum'] / stats['count']
            std_dev = np.sqrt((stats['sum']**2 / stats['count'] - mean**2) / max(stats['count'] - 1, 1))
            feature_vector[2*idx] = mean
            feature_vector[2*idx + 1] = std_dev
        else:
            feature_vector[2*idx] = -1
            feature_vector[2*idx + 1] = -1

    return feature_vector

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process DNA sequences with Chaos Game Representation.')
    parser = argparse.ArgumentParser(description='Process DNA sequences for RTD feature extraction.')
    parser.add_argument('-d', '--Degenerate', choices=['replace', 'remove'], required=True, 
                        help='Specify the action on degenerate nucleotides: replace them or remove them entirely.')
    parser.add_argument('-k', '--Word_Length', type=int, choices=[5, 6, 7], required=True, 
                        help='Define the length of k-mers to be used in RTD calculations.')
    parser.add_argument('-v', '--Data', choices=['SARS-CoV-2', 'HIV'], required=True,
                        help='Specify the virus dataset.')
    args = parser.parse_args()
    
    print(f"Running RTD with {args.Degenerate} option at k={args.Word_Length}.")

    sequence_df = load_sequences(args)

    # Set the k-mer length as specified by the user
    k = args.Word_Length

    # Precompute a dictionary with statistics for all possible k-mers
    kmer_stats = precompute_kmer_dict(k)
    
    RTD_data = np.zeros((len(sequence_df), len(kmer_stats)*2), dtype=np.float32)

    # Fill in the preallocated array
    for i, sequence in tqdm(enumerate(sequence_df["Sequence"]), total=len(sequence_df), desc="Computing RTD feature vectors"):
        RTD_data[i, :] = compute_rtd_feature_vector(sequence, k, copy.deepcopy(kmer_stats))
        
    # Convert the list of RTD feature vectors into a DataFrame
    RTD_data = pd.DataFrame(RTD_data)
    
    targets = sequence_df['Target'].to_list()
    train_labels = sequence_df['Train'].to_list()
    save_features_to_parquet(RTD_data, args, targets, train_labels)
    
if __name__ == "__main__":
    main()
