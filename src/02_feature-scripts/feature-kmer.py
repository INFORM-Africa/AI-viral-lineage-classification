import argparse
import pandas as pd
import numpy as np
from collections import Counter
from itertools import product
from tqdm import tqdm
from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides

def get_kmers(sequence, k=3):
    """
    Generate k-mers for a given DNA sequence.

    Parameters:
    - sequence (str): The DNA sequence to be processed.
    - k (int): The length of k-mers to generate.

    Returns:
    - list: A list of k-mer strings extracted from the sequence.
    """
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def generate_kmer_features(sequences, k=3, alphabet='ACGT'):
    """
    Generate k-mer feature counts for a list of DNA sequences.

    Parameters:
    - sequences (list): A list of DNA sequences.
    - k (int): The length of k-mers to generate.
    - alphabet (str): The alphabet to use for generating all possible k-mers.

    Returns:
    - np.array: An array of k-mer feature vectors for each sequence.
    """
    # Generate all possible k-mers from the given alphabet
    all_kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    
    feature_vectors = []
    
    # Process each sequence and generate its feature vector
    for seq in tqdm(sequences, desc=f"Generating k-mer features with k={k}"):
        # Generate k-mer counts for the current sequence
        kmer_counts = Counter(get_kmers(seq, k))
        
        # Initialize a feature vector with zeros for all possible k-mers
        feature_vector = np.zeros(len(all_kmers))
        
        # Update the feature vector with counts from the current sequence
        for idx, kmer in enumerate(all_kmers):
            feature_vector[idx] = kmer_counts.get(kmer, 0)
        
        # Append the feature vector to the list
        feature_vectors.append(feature_vector)
    
    return np.array(feature_vectors)

def main():
    parser = argparse.ArgumentParser(description='Generate k-mer features from DNA sequences.')
    parser.add_argument('-d', '--Degenerate', choices=['Replace', 'Remove'], required=True, help='Whether to replace or remove degenerate nucleotides.')
    parser.add_argument('-k', '--Word_Length', type=int, choices=[5, 6, 7], required=True, help='Length of k-mers.')
    args = parser.parse_args()

    print(f"Running k-mers with {args.Degenerate} option at k={args.Word_Length}.")
    print("Loading Data...")
    
    # Load DNA sequence data
    data = pd.read_parquet('../../data/processed/cov-19.parquet', engine='pyarrow')

    # Process degenerate nucleotides based on user input
    if args.Degenerate == 'Replace':
        print("Replacing degenerate nucleotides...")
        data = replace_degenerate_nucleotides(data)
    elif args.Degenerate == 'Remove':
        print("Removing degenerate nucleotides...")
        data = remove_degenerate_nucleotides(data)

    print("Generating k-mer features...")
    # Generate k-mer features
    kmer_features = generate_kmer_features(data['Sequence'].tolist(), k=args.Word_Length)

    # Create DataFrame from k-mer features and add target labels
    kmer_df = pd.DataFrame(kmer_features)
    kmer_df["Target"] = data["Lineage"].tolist()
    kmer_df["Train"] = data["Train"].tolist()

    # Save k-mer features to a Parquet file
    output_filename = f'../../data/features/{args.Word_Length}-mer_{args.Degenerate.lower()}.parquet'
    kmer_df.to_parquet(output_filename, engine='pyarrow')
    print(f"K-mer features saved to {output_filename}")

if __name__ == "__main__":
    main()