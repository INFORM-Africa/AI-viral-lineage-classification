import argparse
import numpy as np
import pandas as pd
from numba import njit
from pydivsufsort import divsufsort, kasai
import numba
import concurrent.futures
import threading
from threading import Lock
from tqdm import tqdm
from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides

lock = Lock()

def random_undersample(data_df, max_samples_per_class=1, random_state=42):

    undersampled_data = []

    for class_value, group in data_df.groupby('Lineage'):
        if len(group) > max_samples_per_class:
            undersampled_group = group.sample(n=max_samples_per_class, random_state=random_state)
        else:
            undersampled_group = group
        undersampled_data.append(undersampled_group)

    undersampled_data_df = pd.concat(undersampled_data)
    undersampled_data_df = undersampled_data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return undersampled_data_df

def safe_divsufsort_kasai(S):
    """
    Compute the suffix array of a given string in a thread-safe manner.

    Parameters:
    - S (str): The input string for which to compute the suffix array.

    Returns:
    - SA (np.array): The computed suffix array.
    """
    with lock:
        # Ensure that divsufsort is called in a thread-safe way
        SA = divsufsort(S)  # Assuming 'get_SA' is a pre-defined function for divsufsort
    return SA

@numba.njit
def process_first_loop(LCP, same_seq, n):
    """
    Process the first loop to compute a part of the result using the LCP array.

    Parameters:
    - LCP (np.array): The LCP array of the input string.
    - same_seq (np.array): A boolean array indicating if consecutive elements belong to the same sequence.
    - n (int): The length of the input string.

    Returns:
    - f (np.array): The result array after processing the first loop.
    """
    f = np.zeros(n, dtype=np.int16)  # Initialize the result array with zeros
    min_val = 0  # Initialize min to 0
    for i in range(1, n-2):
        if same_seq[i]:
            if LCP[i+1] < min_val:
                min_val = LCP[i+1]
            f[i + 1] = min_val
        else:
            min_val = LCP[i+1]
            f[i + 1] = LCP[i+1]
    return f

@numba.njit
def process_second_loop(LCP, same_seq, n):
    """
    Process the second loop to refine the result using the LCP array.

    Parameters:
    - LCP (np.array): The LCP array of the input string.
    - same_seq (np.array): A boolean array indicating if consecutive elements belong to the same sequence.
    - n (int): The length of the input string.

    Returns:
    - f (np.array): The refined result array after processing the second loop.
    """
    f = np.zeros(n, dtype=np.int16)  # Re-initialize the result array with zeros
    min_val = 0  # Re-initialize min to 0 for the second loop
    for i in range(n-1, 1, -1):
        if same_seq[i-1]:  # Adjusted index for same_seq
            if LCP[i] < min_val:
                min_val = LCP[i]
            f[i - 1] = max(min_val, f[i - 1])
        else:
            min_val = LCP[i]
            f[i - 1] = max(min_val, f[i - 1])
    return f

def acs(A, B):
    """
    Compute the Average Common Substring (ACS) score between two sequences.

    Parameters:
    - A (str): The first sequence.
    - B (str): The second sequence.

    Returns:
    - d_ACS (float): The computed ACS score.
    """
    
    S = f"{A}${B}"  # Concatenate A and B with a delimiter for processing

    SA = safe_divsufsort_kasai(S)  # Compute the suffix array in a thread-safe manner
    
    LCP = kasai(S, SA)  # Assuming 'kasai' is a pre-defined function for computing the LCP array
    LCP = np.append(-1, LCP[:-1])  # Adjust LCP array as needed for processing

    n = len(S)
    mid = len(A) + 1  # Compute the midpoint index

    # Compute same_seq array to identify elements belonging to the same sequence
    is_A = SA < mid
    is_A_shifted = np.roll(is_A, -1)
    same_seq = is_A & is_A_shifted
    same_seq |= (~is_A) & np.roll(~is_A, -1)    
    same_seq[-1] = False  # Ensure the last element is marked as False

    f1 = process_first_loop(LCP, same_seq, n)  # Process the first loop to compute a part of the result

    f2 = process_second_loop(LCP, same_seq, n)  # Process the second loop to refine the result

    f = np.maximum(f1, f2)
        
    A_scores = f[is_A]
    B_scores = f[~is_A]
    
    d_AB = np.log(len(B))/np.mean(A_scores) - 2*np.log(len(A))/len(A)
    d_BA = np.log(len(A))/np.mean(B_scores) - 2*np.log(len(B))/len(B)
    
    d_ACS = (d_AB + d_BA)/2
    return d_ACS

def main():

    parser = argparse.ArgumentParser(description='Process DNA sequences using Chaos Game Representation (CGR).')
    parser.add_argument('-d', '--Degenerate', choices=['Replace', 'Remove'], required=True,
                        help='Specify how to handle degenerate nucleotides: Replace or Remove.')
    args = parser.parse_args()
    
    print(f"Running ACS with {args.Degenerate}")
    print("Loading Data...")
    
    data = pd.read_parquet('../../data/processed/cov-19.parquet', engine='pyarrow')

    
    # Preprocessing the data based on the degenerate nucleotides option.
    if args.Degenerate == 'Replace':
        print("Replacing degenerate nucleotides...")
        data = replace_degenerate_nucleotides(data)
    elif args.Degenerate == 'Remove':
        print("Removing degenerate nucleotides...")
        data = remove_degenerate_nucleotides(data)
    
    #Select Comparison Sequences
    comparison_sequences = random_undersample(data)["Sequence"]

    print("Generating ACS Distance Matrix")

    def calculate_acs_for_genome(genome):
        scores = np.zeros(len(comparison_sequences))
        for i, comp_seq in enumerate(comparison_sequences):
            scores[i] = acs(genome, comp_seq)  # Assuming 'acs' is a previously defined function
        return scores

    acs_distances = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all genomes for processing, where each genome will be processed in parallel
        futures = {executor.submit(calculate_acs_for_genome, genome): genome for genome in data["Sequence"]}

        # Wrap concurrent.futures.as_completed(futures) with tqdm for a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Computing ACS distances"):
            acs_distances.append(future.result())

    acs_data = pd.DataFrame(acs_distances)
    acs_data["Target"] = data["Lineage"].tolist()
    acs_data["Train"] = data["Train"].tolist()
    acs_data.to_parquet(f'../../data/features/ACS_{args.Degenerate.lower()}.parquet', engine='pyarrow')

if __name__ == "__main__":
    main()