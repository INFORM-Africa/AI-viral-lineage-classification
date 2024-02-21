import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides

def generate_chaos_game_representation(sequence, resolution):
    """
    Generates a Chaos Game Representation (CGR) of a DNA sequence.

    Parameters:
    - sequence (str): The DNA sequence to be processed.
    - resolution (int): The resolution of the CGR image, determining its size.

    Returns:
    - np.array: A 2D array representing the CGR image.
    """
    # Mapping nucleotides to their respective points in the unit square.
    nucleotide_mapping = {'A': (0, 0), 'C': (0, 1), 'G': (1, 1), 'T': (1, 0)}
    image = np.zeros((resolution, resolution))

    # Starting point in the center of the square.
    x, y = 0.5, 0.5
    scale = resolution - 1

    # Iterating through each nucleotide in the sequence.
    for nucleotide in sequence:
        if nucleotide in nucleotide_mapping:
            corner_x, corner_y = nucleotide_mapping[nucleotide]
            x = (x + corner_x) / 2
            y = (y + corner_y) / 2

            # Mapping the continuous position to discrete pixel coordinates.
            ix, iy = int(x * scale), int(y * scale)
            image[iy, ix] += 1

    return image

def main():
    parser = argparse.ArgumentParser(description='Process DNA sequences using Chaos Game Representation (CGR).')
    parser.add_argument('-d', '--Degenerate', choices=['Replace', 'Remove'], required=True,
                        help='Specify how to handle degenerate nucleotides: Replace or Remove.')
    parser.add_argument('-r', '--Resolution', type=int, choices=[64, 128], required=True,
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

    # Generating FCGR for each sequence in the dataset.
    cgr_images = [generate_chaos_game_representation(sequence, args.Resolution) for sequence in tqdm(data['Sequence'], desc="Generating FCGR Images")]
    cgr_flattened = [image.flatten() for image in cgr_images]
    chaos_data = pd.DataFrame(cgr_flattened)
    chaos_data["Target"] = data["Lineage"].tolist()
    chaos_data["Train"] = data["Train"].tolist()

    print("Saving processed data...")
    output_filename = f'../../data/features/FCGR_{args.Degenerate.lower()}_{args.Resolution}.parquet'
    chaos_data.to_parquet(output_filename, engine='pyarrow')

if __name__ == "__main__":
    main()