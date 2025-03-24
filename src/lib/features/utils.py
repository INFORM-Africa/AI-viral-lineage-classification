import pandas as pd
import numpy as np
import random
import os

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


def load_sequences(args):
    """
    Load sequence data from a parquet file and preprocess it based on the degenerate nucleotides option.

    Args:
        args: An object with attributes 'Data' and 'Degenerate' which specify the dataset and the 
              degenerate nucleotides handling method respectively.

    Returns:
        A pandas DataFrame with the loaded and preprocessed sequence data.
    """
    # Load the data based on the specified dataset
    if args.Data == 'SARS-CoV-2':
        data_path = '../../data/sequences/SARS-CoV-2.parquet'
    else:
        data_path = '../../data/sequences/HIV.parquet'
    
    data = pd.read_parquet(data_path, engine='pyarrow')

    # Preprocess the data based on the specified degenerate nucleotides option
    if args.Degenerate == 'replace':
        data = replace_degenerate_nucleotides(data)
    elif args.Degenerate == 'remove':
        data = remove_degenerate_nucleotides(data)

    return data

def save_features_to_parquet(feature_data, args, targets, train_labels):
    """
    Converts the given data to a pandas DataFrame (if necessary), appends the 'Target' and 'Train' columns, 
    and saves the DataFrame to a parquet file.

    Args:
        data: NumPy array or pandas DataFrame containing the feature data.
        filename: The base filename to save the data to.
        args: An object containing attributes used to construct the file path and additional data columns.
    """
    # Convert the numpy array to a pandas DataFrame if it's not already a DataFrame
    if not isinstance(feature_data, pd.DataFrame):
        feature_data = pd.DataFrame(feature_data)

    # Append the 'Target' and 'Train' columns from the original data
    feature_data["Target"] = targets
    feature_data["Train"] = train_labels

    # Construct the output directory and filename
    output_dir = f'../../data/features/'
    output_filename = f'{output_dir}{args.Data}.parquet'
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure the column names are string type, then save to a Parquet file
    feature_data.columns = feature_data.columns.map(str)
    feature_data.to_parquet(output_filename, engine='pyarrow')
    
    print(f"Features saved to {output_filename}")
