import pandas as pd
import numpy as np
import mmh3
from collections import defaultdict
from tqdm import tqdm

def generate_kmers(sequence, k):
    """
    Generate all possible k-mers from a given DNA sequence.

    Parameters:
    - sequence (str): The DNA sequence from which k-mers are to be generated.
    - k (int): The length of each k-mer.

    Yields:
    - str: Subsequences (k-mers) of length 'k' from the given sequence.
    """
    for i in range(len(sequence) - k + 1):
        yield sequence[i:i + k]

def create_mash_sketch(sequence, k, sketch_size, coverage_threshold=1):
    """
    Create a MinHash sketch (similar to Mash) for a given sequence by selecting the smallest hashes of its k-mers.

    This function employs a two-pass approach, initially counting all k-mer occurrences and then filtering based on
    a coverage threshold to only consider k-mers that appear at least a specified number of times.

    Parameters:
    - sequence (str): The DNA sequence to be sketched.
    - k (int): The length of k-mers to consider for hashing.
    - sketch_size (int): The number of hashes to retain in the sketch, effectively the sketch's "resolution".
    - coverage_threshold (int, optional): The minimum number of occurrences for a k-mer to be included in the sketch. Defaults to 1.

    Returns:
    - np.array: An array of the smallest 'sketch_size' hash values from the filtered k-mers, representing the sketch.
    """
    kmer_counts = defaultdict(int)

    # First pass: Count all k-mer occurrences
    for kmer in generate_kmers(sequence, k):
        kmer_hash = mmh3.hash(kmer)
        kmer_counts[kmer_hash] += 1
    
    # Filter k-mers based on coverage threshold and sort by hash value
    filtered_hashes = [kmer_hash for kmer_hash, count in kmer_counts.items() if count >= coverage_threshold]
    sketch = sorted(filtered_hashes)[:sketch_size]

    return np.array(sketch)

def main():
    
    parser = argparse.ArgumentParser(description='Process DNA sequences with Chaos Game Representation.')
    parser.add_argument('-d', '--Degenerate', choices=['Replace', 'Remove'], required=True, help='Whether to replace or remove degenerate nucleotides.')
    parser.add_argument('-k', '--Word_Length', type=int, choices=[13], required=True, help='Length of k-mers.')
    args = parser.parse_args()
    
    print(f"Running Mash Sketch with {args.Degenerate} option at k={args.Word_Length}.")
    print("Loading data...")

    data = pd.read_parquet('../../data/processed/cov-19.parquet', engine='pyarrow')
    
    # Process degenerate nucleotides based on user input
    if args.Degenerate == 'Replace':
        print("Replacing degenerate nucleotides...")
        data = replace_deg(data)
    elif args.Degenerate == 'Remove':
        print("Removing degenerate nucleotides...")
        data = remove_deg(data)

    k = args.Word_Length
    sketch_size = 1000
    
    print("Generating MASH sketches...")

    sketch_array = []
    for genome in tqdm(data["Sequence"], desc="Creating Mash Sketches"):
        sketch_array.append(create_optimized_mash_sketch(genome, k, sketch_size))

    mash_data = pd.DataFrame(sketch_array)
    mash_data["Target"] = data["Lineage"].tolist()
    mash_data["Train"] = data["Train"].tolist()
    mash_data.to_parquet(f'../../data/features/MASH_sketch{args.Degenerate}.parquet', engine='pyarrow')

if __name__ == "__main__":
    main()

