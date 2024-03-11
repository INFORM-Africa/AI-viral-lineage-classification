import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides
from numba import jit
import os

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
    parser.add_argument('-d', '--Degenerate', choices=['replace', 'remove'], required=True,
                        help='Specify how to handle degenerate nucleotides: Replace or Remove.')
    parser.add_argument('-r', '--Resolution', type=int, choices=[64, 128, 256], required=True,
                        help='Resolution of the CGR image (e.g., 64x64 or 128x128 pixels).')
    parser.add_argument('-v', '--Data', choices=['SARS', 'HIV'], required=True,
                        help='Specify the virus dataset.')
    args = parser.parse_args()

    print(f"Running CGR with {args.Degenerate} option at {args.Resolution}x{args.Resolution} resolution.")

    # Loading DNA sequence data.
    if args.Data == 'SARS':
        data = pd.read_parquet('../../../data/processed/SARS-CoV-2.parquet', engine='pyarrow')
    else:
        data = pd.read_parquet('../../../data/processed/HIV.parquet', engine='pyarrow')

    # Preprocessing the data based on the degenerate nucleotides option.
    if args.Degenerate == 'replace':
        data = replace_degenerate_nucleotides(data)
    elif args.Degenerate == 'remove':
        data = remove_degenerate_nucleotides(data)
    
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

    output_dir = f'../../../data/features/{args.Data}'
    output_filename = f'{output_dir}/FCGR_{args.Degenerate}_{args.Resolution}.parquet'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chaos_data.columns = chaos_data.columns.map(str)
    chaos_data.to_parquet(output_filename, engine='pyarrow')
    print(f"FCGR features saved to {output_filename}")

if __name__ == "__main__":
    main()