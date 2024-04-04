import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import numba
import argparse
import time
from numba import jit
from tqdm import tqdm
import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar

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

def calculate_mash_distance(sketch1, sketch2, k):
    """
    Calculate the Mash distance between two MinHash sketches.

    :param sketch1: First MinHash sketch as a numpy array.
    :param sketch2: Second MinHash sketch as a numpy array.
    :param k: k-mer size used to create the MinHash sketches.
    :return: The Mash distance.
    """
    start = time.time()
    
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
    return [calculate_mash_distance(sketch1, sketch2, k) for sketch2 in pair_data]

def calculate_distance_matrix(data, pair_data, k):
    """
    Calculate a matrix of Mash distances between two arrays of sketches using Dask for parallel computation.

    :param data: First array of MinHash sketches.
    :param pair_data: Second array of MinHash sketches to which to compare the first array.
    :param k: k-mer size used to create the MinHash sketches.
    :return: A Dask array of Mash distances.
    """
    delayed_rows = []  # Initialize outside the loop

    # Create a list of delayed computations for each row
    for sketch1 in data:
        delayed_row = delayed(calculate_distances_for_sketch)(sketch1, pair_data, k)
        delayed_rows.append(delayed_row)

    progress_bar = ProgressBar()
    progress_bar.register()

    # Compute all rows in parallel and concatenate the results
    with progress_bar:
        distance_matrix = compute(*delayed_rows, scheduler='processes')

    return np.vstack(distance_matrix)

def main():
    parser = argparse.ArgumentParser(description='Process DNA sequences using Chaos Game Representation (CGR).')
    parser.add_argument('-f', '--File', type=str, required=True, help='Specify DSP Signal Feature File')
    parser.add_argument('-k', '--Word_Length', type=int, required=True, help='Specify Word Length')
    parser.add_argument('-v', '--Data', choices=['SARS', 'HIV'], required=True,
                            help='Specify the virus dataset.')
    args = parser.parse_args()

    print(f"Running Mash-dist with {args.Degenerate} option at k={args.Word_Length}."
    data = pd.read_parquet(f'../../data/features/{args.Data}/{args.File}.parquet', engine='pyarrow')  # You can use 'fastparquet' as the engine

    targets = data["Target"]
    train = data["Train"]

    pair_data = data[data["Train"] == 0]

    data = data.drop(columns=["Train", "Target"]).to_numpy().astype(np.float32)
    pair_data = pair_data.drop(columns=["Train", "Target"]).to_numpy().astype(np.float32)

    distance_matrix = calculate_distance_matrix(data, pair_data, args.Word_Length)
    distance_df = pd.DataFrame(distance_matrix)
    distance_df["Target"] = targets.tolist()
    distance_df["Train"] = train.tolist()

    distance_df.to_parquet(f'../../data/features/{args.Data}/{args.File}_dist.parquet', engine='pyarrow')

if __name__ == '__main__':
    main()