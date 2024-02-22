import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides
from numba import jit

@jit(nopython=True)
def generate_chaos_game_representation(sequence, resolution, nucleotide_mapping):
    image = np.zeros((resolution, resolution), dtype=np.int16)

    x, y = 0.5, 0.5
    scale = resolution - 1

    for char in sequence:
        if char == 'A':
            index = 0
        elif char == 'C':
            index = 1
        elif char == 'G':
            index = 2
        elif char == 'T':
            index = 3
        else:
            continue  # Skip unknown characters

        corner_x, corner_y = nucleotide_mapping[index]
        x = (x + corner_x) / 2
        y = (y + corner_y) / 2

        ix, iy = int(x * scale), int(y * scale)
        image[iy, ix] += 1

    return image.flatten()


def main():
    parser = argparse.ArgumentParser(description='Process DNA sequences using Chaos Game Representation (CGR).')
    parser.add_argument('-d', '--Degenerate', choices=['Replace', 'Remove'], required=True,
                        help='Specify how to handle degenerate nucleotides: Replace or Remove.')
    parser.add_argument('-r', '--Resolution', type=int, choices=[64, 128, 256], required=True,
                        help='Resolution of the CGR image (e.g., 64x64 or 128x128 pixels).')
    args = parser.parse_args()

    print(f"Running CGR with {args.Degenerate} option at {args.Resolution}x{args.Resolution} resolution.")
    print("Loading data...")

    # Loading DNA sequence data.
    data = pd.read_parquet('../../data/processed/cov-19.parquet', engine='pyarrow')

    # Preprocessing the data based on the degenerate nucleotides option.
    if args.Degenerate == 'Replace':
        print("Replacing degenerate nucleotides...")
        data = replace_degenerate_nucleotides(data)
    elif args.Degenerate == 'Remove':
        print("Removing degenerate nucleotides...")
        data = remove_degenerate_nucleotides(data)

    print("Generating Chaos Game Representation...")
    
    # Preallocate numpy array
    chaos_data = np.zeros((len(data), args.Resolution * args.Resolution), dtype=np.int16)
    nucleotide_mapping = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)  # A, C, G, T

    # Fill in the preallocated array
    for i, sequence in tqdm(enumerate(data["Sequence"]), total=len(data), desc="Generating CGR"):
        chaos_data[i, :] = generate_chaos_game_representation(sequence, args.Resolution, nucleotide_mapping)

    # Convert the numpy array to a pandas DataFrame if needed
    chaos_data = pd.DataFrame(chaos_data)
    chaos_data["Target"] = data["Lineage"].tolist()
    chaos_data["Train"] = data["Train"].tolist()

    print("Saving processed data...")
    output_filename = f'../../data/features/FCGR_{args.Degenerate.lower()}_{args.Resolution}.parquet'
    chaos_data.to_parquet(output_filename, engine='pyarrow')

if __name__ == "__main__":
    main()