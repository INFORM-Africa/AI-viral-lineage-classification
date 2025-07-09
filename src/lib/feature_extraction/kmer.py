import numpy as np
from collections import Counter
from itertools import product

def get_kmers(sequence, k):
    """
    Generate k-mers for a given DNA sequence.

    Parameters:
    - sequence (str): The DNA sequence to be processed.
    - k (int): The length of k-mers to generate.

    Returns:
    - list: A list of k-mer strings extracted from the sequence.
    """
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def kmer_representation(partition, k, alphabet='ACGT'):
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
          
    for seq in partition['Sequence']:
        # Generate k-mer counts for the current sequence
        kmer_counts = Counter(get_kmers(seq, k))
        
        # Initialize a feature vector with zeros for all possible k-mers
        feature_vector = np.zeros(len(all_kmers), dtype=np.uint16)
        
        # Update the feature vector with counts from the current sequence
        for idx, kmer in enumerate(all_kmers):
            feature_vector[idx] = kmer_counts.get(kmer, 0)
        
        # Append the feature vector to the list
        feature_vectors.append(feature_vector)

    return np.array(feature_vectors)