import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import numba
import argparse

parser = argparse.ArgumentParser(description='Process DNA sequences using Chaos Game Representation (CGR).')
parser.add_argument('-f', '--File', type=str, required=True, help='Specify DSP Signal Feature File')
parser.add_argument('-v', '--Data', choices=['SARS-CoV-2', 'HIV'], required=True,
                        help='Specify the virus dataset.')

args = parser.parse_args()

data = pd.read_parquet(f'../../data/features/{args.Data}/{args.File}.parquet', engine='pyarrow')  # You can use 'fastparquet' as the engine

targets = data["Target"]
train = data["Train"]

pair_data = data[data["Train"] == 0]

data = data.drop(columns=["Train", "Target"]).to_numpy().astype(np.float32)
pair_data = pair_data.drop(columns=["Train", "Target"]).to_numpy().astype(np.float32)

print("Standardizing Data")
# Function to standardize data
def standardize(data):
    mean = np.mean(data, axis=1, keepdims=True).astype(np.float32)  # Use higher precision for mean and std calculations
    std = np.std(data, axis=1, keepdims=True, ddof=1).astype(np.float32)
    return ((data.astype(np.float32) - mean) / std).astype(np.float32)  # Convert back to float16 after standardization

# Standardize the data and pair_data arrays
data = standardize(data)
pair_data = standardize(pair_data)

print("Computing Dot Product")
@numba.njit
def fast_dot_product(data, pair_data):
    # Ensure the input is float32 for the dot product to minimize precision issues
    result = np.dot(data.astype(np.float32), pair_data.T.astype(np.float32))
    return result.astype(np.float32)  # Convert back to float16 if needed

# Compute the dot product using Numba
dot_product = fast_dot_product(data, pair_data)

n = data.shape[1]  # Number of features
correlation_matrix = (dot_product / (n - 1) + 1) / 2  # Normalize the correlation matrix to be between 0 and 1
correlation_matrix = (1 - correlation_matrix) / 2
correlation_matrix = pd.DataFrame(correlation_matrix)

correlation_matrix["Target"] = targets.to_list()
correlation_matrix["Train"] = train.to_list()

correlation_matrix.to_parquet(f'../../data/features/{args.Data}/{args.File}_dist.parquet', engine='pyarrow')
