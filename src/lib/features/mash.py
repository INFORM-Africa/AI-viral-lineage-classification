import numpy as np
import mmh3
from collections import defaultdict
from numba import jit

def generate_kmers(sequence, k):
    """
    Generate all possible k-mers from a given DNA sequence.

    Parameters:
    - sequence (str): The DNA sequence from which k-mers are to be generated.
    - k (int): The length of each k-mer.

    Yields:
    - str: Subsequences (k-mers) of length 'k' from the given sequence.
    """
    for i in range(len(sequence) - k + 1):
        yield sequence[i:i + k]

def create_mash_sketch(sequence, k, sketch_size, coverage_threshold=1):
    """
    Create a MinHash sketch (similar to Mash) for a given sequence by selecting the smallest hashes of its k-mers.

    This function employs a two-pass approach, initially counting all k-mer occurrences and then filtering based on
    a coverage threshold to only consider k-mers that appear at least a specified number of times.

    Parameters:
    - sequence (str): The DNA sequence to be sketched.
    - k (int): The length of k-mers to consider for hashing.
    - sketch_size (int): The number of hashes to retain in the sketch, effectively the sketch's "resolution".
    - coverage_threshold (int, optional): The minimum number of occurrences for a k-mer to be included in the sketch. Defaults to 1.

    Returns:
    - np.array: An array of the smallest 'sketch_size' hash values from the filtered k-mers, representing the sketch.
    """
    kmer_counts = defaultdict(int)

    # First pass: Count all k-mer occurrences
    for kmer in generate_kmers(sequence, k):
        kmer_hash = mmh3.hash(kmer)
        kmer_counts[kmer_hash] += 1
    
    # Filter k-mers based on coverage threshold and sort by hash value
    filtered_hashes = [kmer_hash for kmer_hash, count in kmer_counts.items() if count >= coverage_threshold]
    sketch = sorted(filtered_hashes)[:sketch_size]

    return np.array(sketch)

@jit(nopython=True)
def calculate_jaccard_index(sketch1, sketch2):
    """
    Efficiently calculate the Jaccard index between two sorted MinHash sketches.
    
    :param sketch1: First sorted MinHash sketch as a numpy array.
    :param sketch2: Second sorted MinHash sketch as a numpy array.
    :return: The estimated Jaccard index.
    """
    shared_hashes = 0  # Intersection
    total_hashes = 0   # Union
    i, j = 0, 0
    
    while i < len(sketch1) and j < len(sketch2):
        if sketch1[i] == sketch2[j]:
            shared_hashes += 1
            i += 1
            j += 1
        elif sketch1[i] < sketch2[j]:
            i += 1
        else:
            j += 1
        total_hashes += 1

    # Include any remaining hashes from both sketches
    total_hashes += len(sketch1[i:]) + len(sketch2[j:])

    # Calculate the Jaccard index
    jaccard_index = shared_hashes / total_hashes if total_hashes > 0 else 0
    
    return jaccard_index

def calculate_distance(sketch1, sketch2, k):
    """
    Calculate the Mash distance between two MinHash sketches.

    :param sketch1: First MinHash sketch as a numpy array.
    :param sketch2: Second MinHash sketch as a numpy array.
    :param k: k-mer size used to create the MinHash sketches.
    :return: The Mash distance.
    """

    jaccard_estimate = calculate_jaccard_index(sketch1, sketch2)
    # Calculate the Mash distance using the formula
    # Guard against log(0) by maxing jaccard_estimate with a very small number
    jaccard_estimate = max(jaccard_estimate, 1e-10)
    mash_distance = - (1 / k) * np.log((2 * jaccard_estimate) / (1 + jaccard_estimate))

    return mash_distance

def calculate_distances_for_sketch(sketch1, pair_data, k):
    """
    Calculate the Mash distances between a single sketch and an array of sketches.

    :param sketch1: The single MinHash sketch to compare against pair_data.
    :param pair_data: Array of MinHash sketches to compare with sketch1.
    :param k: k-mer size used to create the MinHash sketches.
    :return: A list of Mash distances.
    """
    return [calculate_distance(sketch1, sketch2, k) for sketch2 in pair_data]

def mash_distance(partition_sketches, candidate_sketches, k):
    """
    Calculate a matrix of Mash distances between two arrays of sketches using Dask for parallel computation.

    :param data: First array of MinHash sketches.
    :param pair_data: Second array of MinHash sketches to which to compare the first array.
    :param k: k-mer size used to create the MinHash sketches.
    :return: A Dask array of Mash distances.
    """
    distance_matrix = []  # Initialize outside the loop
    
    # Create a list of delayed computations for each row
    for partition_sketch in partition_sketches:
        distance_matrix.append(calculate_distances_for_sketch(partition_sketch, candidate_sketches, k))

    return np.vstack(np.abs(distance_matrix)).astype(np.float16)

def mash_sketch_representation(partition, k, sketch_size):
    partition = partition.reset_index()

    local_features = np.zeros((len(partition), sketch_size))
    for i, sequence in enumerate(partition['Sequence']):
        local_features[i, :] = create_mash_sketch(sequence, k, sketch_size)
    
    return local_features
