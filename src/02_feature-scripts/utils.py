import pandas as pd
import numpy as np

def replace_degenerate_nucleotides(genomes):
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

def remove_degenerate_nucleotides(genomes):
    genomes['Sequence'] = genomes['Sequence'].str.replace('[^ACTG]', '', regex=True)
    return genomes
