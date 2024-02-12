import numpy as np
import pandas as pd
import random
import re

def get_features(feature_strat, num_rep=None):
    data = None
    if feature_strat == "kmer":
        data = pd.read_csv('../../data/features/k-mer_3.csv', index_col=0)
    elif feature_strat == "choas":
        data = pd.read_csv('../../data/features/chaos.csv', index_col=0)
    elif feature_strat == "dsp":
        genomes = pd.read_csv('../../data/genomes_replaced.csv', index_col=0)
        data = get_real_rep(genomes)
    return data

def get_real_rep(data):
    real_map = {'A': -1.5, 'T': 1.5, 'C': 0.5, 'G': -0.5}

    # Determine the median length of the sequences
    median_len = int(data['Sequence'].str.len().median())

    # Initialize an empty 2D NumPy array with shape (number of sequences, median_len) and type float16
    numeric_sequences = np.zeros((len(data), median_len), dtype=np.float16)

    # Iterate over the DataFrame and fill the array
    for i, sequence in enumerate(data['Sequence']):
        # Truncate the sequence to the median length if necessary
        truncated_sequence = sequence[:median_len]
        # Map each base in the truncated sequence to its numeric representation, defaulting to 0 for non-ATCG characters
        numeric_sequence = np.array([real_map[base] if base in real_map else 0 for base in truncated_sequence], dtype=np.float16)
        numeric_sequences[i, :len(numeric_sequence)] = numeric_sequence

    return numeric_sequences

def get_eiip_rep(data):
    eiip_map = {'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335}
    data['Sequence'] = data['Sequence'].apply(lambda x: np.array([eiip_map[base] for base in x]))
    return data

def get_integer_rep(data):
    integer_map = {'T': 0, 'C': 1, 'A': 2, 'G': 3}
    data['Sequence'] = data['Sequence'].apply(lambda x: np.array([integer_map[base] for base in x]))
    return data

def get_atomic_rep(data):
    atomic_map = {'A': 70, 'C': 58, 'G': 78, 'T': 66}
    data['Sequence'] = data['Sequence'].apply(lambda x: np.array([atomic_map[base] for base in x]))
    return data

def get_paired_rep(data):
    paired_map = {'A': 1, 'T': 1, 'C': -1, 'G': -1}
    data['Sequence'] = data['Sequence'].apply(lambda x: np.array([paired_map[base] for base in x]))
    return data

def get_justA_rep(data):
    data['Sequence'] = data['Sequence'].apply(lambda x: np.array([1 if base == 'A' else 0 for base in x]))
    return data

def get_numerical_rep(data, num_rep):
    if num_rep == 'real':
        return get_real_rep(data)
    elif num_rep == 'eiip':
        return get_eiip_rep(data)
    elif num_rep == 'integer':
        return get_integer_rep(data)
    elif num_rep == 'atomic':
        return get_atomic_rep(data)
    elif num_rep == 'paired':
        return get_paired_rep(data)
    elif num_rep == 'justA':
        return get_justA_rep(data)
    else:
        raise ValueError("Invalid numeric representation type")

# Define the function again after the reset
def replace_deg(genomes):
    # Define the mapping of degenerate characters to their possible bases
    degenerate_mapping = {
        'W': ['A', 'T'],
        'S': ['C', 'G'],
        'M': ['A', 'C'],
        'K': ['G', 'T'],
        'R': ['A', 'G'],
        'Y': ['C', 'T'],
        'B': ['C', 'G', 'T'],
        'D': ['A', 'G', 'T'],
        'H': ['A', 'C', 'T'],
        'V': ['A', 'C', 'G'],
        'N': ['A', 'C', 'T', 'G']
    }

    # Function to replace a single degenerate character with a random possible base
    def replace_char(char):
        if char in degenerate_mapping:
            return random.choice(degenerate_mapping[char])
        else:
            return char

    # Function to replace all degenerate characters in a sequence
    def replace_sequence(sequence):
        return ''.join(replace_char(char) for char in sequence)

    # Replace the sequences in the dataframe
    genomes['Sequence'] = genomes['Sequence'].apply(replace_sequence)

    return genomes