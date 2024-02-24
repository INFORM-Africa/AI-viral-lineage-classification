import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from itertools import product
from joblib import Parallel, delayed
from collections import Counter
from itertools import product
from collections import defaultdict
import time
from dask.diagnostics import ProgressBar
import dask.array as da
from dask import delayed, compute
import numpy as np
from tqdm import tqdm
import random
import argparse

from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides


def generate_kmers(k, alphabet='ACGT'):
    """Generate all possible k-mers from the given alphabet."""
    return [''.join(p) for p in product(alphabet, repeat=k)]

def find_spaced_words(sequence, pattern, pattern_indices, k):
    start = time.time()
    kmer_counts = defaultdict(int)  # Using defaultdict for faster updates
    all_kmers = generate_kmers(k)
    for i in range(len(sequence) - len(pattern) + 1):
    # Preallocate list to avoid list comprehension overhead
        spaced_word_chars = [None] * len(pattern_indices)
        for index, j in enumerate(pattern_indices):
            spaced_word_chars[index] = sequence[i + j]
        spaced_word = ''.join(spaced_word_chars)
        kmer_counts[spaced_word] += 1
    
    # Only include k-mers present in all_kmers to match original function's behavior
    feature_vector = np.array([kmer_counts[kmer] for kmer in all_kmers])
    return feature_vector

def generate_random_pattern(k, l):
    one_positions = random.sample(range(2, l + k + 1), k - 1)
    one_positions.append(1)
    one_positions.sort()
    
    # Create the pattern, filling in '0's where there isn't a '1'
    pattern = ''
    for i in range(1, max(one_positions) + 1):
        if i in one_positions:
            pattern += '1'
        else:
            pattern += '0'
    
    return pattern

def main():
    parser = argparse.ArgumentParser(description='Process DNA sequences with Chaos Game Representation.')
    parser = argparse.ArgumentParser(description='Process DNA sequences for RTD feature extraction.')
    parser.add_argument('-d', '--Degenerate', choices=['Replace', 'Remove'], required=True, 
                        help='Specify the action on degenerate nucleotides: replace them or remove them entirely.')
    parser.add_argument('-k', '--Word_Length', type=int, choices=[5, 6, 7], required=True, 
                        help='Define the length of k-mers to be used in RTD calculations.')
    args = parser.parse_args()

    data = pd.read_parquet('../../data/processed/cov-19.parquet', engine='pyarrow')  # You can use 'fastparquet' as the engine
    data

    # Process degenerate nucleotides based on the user's choice
    if args.Degenerate == 'Replace':
        print("Replacing degenerate nucleotides...")
        data = replace_degenerate_nucleotides(data)  # Replace degenerate nucleotides as per a predefined scheme
    elif args.Degenerate == 'Remove':
        print("Removing degenerate nucleotides...")
        data = remove_degenerate_nucleotides(data)  # Remove any degenerate nucleotides from the sequences

    # Example usage
    k = args.Word_Length
    l = 30  # Length of the pattern
    pattern = generate_random_pattern(k, l)
    pattern_indices = [i for i, char in enumerate(pattern) if char == '1']

    delayed_find_spaced_words = delayed(find_spaced_words)
    tasks = [delayed_find_spaced_words(sequence, pattern, pattern_indices, k) for sequence in data["Sequence"]]

    # Instantiate and register the ProgressBar
    progress_bar = ProgressBar()
    progress_bar.register()

    # Use dask.compute to execute tasks in parallel, now with a progress bar
    with progress_bar:
        feature_vectors = compute(*tasks, scheduler='processes')

    # Stack the resulting feature vectors
    stacked_feature_vectors = np.vstack(feature_vectors)
    spaced_words = pd.DataFrame(stacked_feature_vectors)
    spaced_words["Target"] = data["Lineage"].tolist()
    spaced_words["Train"] = data["Train"].tolist()
    spaced_words.columns = spaced_words.columns.astype(str)

    spaced_words.to_parquet(f'../../data/features/spaced/{k}-spaced_{args.Degenerate.lower()}_{pattern}.parquet', engine='pyarrow')

if __name__ == "__main__":
    main()