import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides, load_sequences, save_features_to_parquet
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
    parser.add_argument('-v', '--Data', choices=['SARS-CoV-2', 'HIV'], required=True,
                        help='Specify the virus dataset.')
    args = parser.parse_args()

    print(f"Running CGR with {args.Degenerate} option at {args.Resolution}x{args.Resolution} resolution.")

    # Loading DNA sequence data.
    sequence_df = load_sequences(args)
    
    # Preallocate numpy array
    FCGR_features = np.zeros((len(sequence_df), args.Resolution * args.Resolution), dtype=np.int16)
    nucleotide_mappings = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)  # A, C, G, T

    # Fill in the preallocated array
    for i, sequence in tqdm(enumerate(sequence_df["Sequence"]), total=len(sequence_df), desc="Generating FCGR Features"):
        FCGR_features[i, :] = generate_chaos_game_representation(sequence, args.Resolution, nucleotide_mappings)
    
    targets = sequence_df['Target'].to_list()
    train_labels = sequence_df['Train'].to_list()
    save_features_to_parquet(FCGR_features, args, targets, train_labels)

if __name__ == "__main__":
    main()