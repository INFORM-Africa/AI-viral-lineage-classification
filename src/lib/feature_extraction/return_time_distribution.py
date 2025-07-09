import numpy as np
from itertools import product
import copy

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

    feature_vector = np.zeros(2 * len(kmer_stats), dtype=np.float16)
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

def return_time_representation(partition, k):
    kmer_stats = precompute_kmer_dict(k)
    local_features = np.zeros((len(partition), len(kmer_stats)*2), dtype=np.float16)
    for i, sequence in enumerate(partition['Sequence']):
        local_features[i, :] = compute_rtd_feature_vector(sequence, k, copy.deepcopy(kmer_stats))
    return local_features
