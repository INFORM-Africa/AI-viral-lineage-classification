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
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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
    feature_vector = np.array([kmer_counts[kmer] for kmer in all_kmers], dtype=np.uint16)
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

def spaced_word_representation(partition, pattern, k):

    partition.reset_index(drop=True, inplace=True)
    pattern_indices = [i for i, char in enumerate(pattern) if char == '1']

    feature_vectors = []

    for sequence in partition['Sequence']:
        # Generate k-mer counts for the current sequence
        feature_vector = find_spaced_words(sequence, pattern, pattern_indices, k)
        
        # Append the feature vector to the list
        feature_vectors.append(feature_vector)

    return np.array(feature_vectors, dtype=np.uint16)
    
