import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from itertools import product
from joblib import Parallel, delayed
from collections import Counter
from itertools import product
from collections import defaultdict
import time
from dask.diagnostics import ProgressBar
import dask.array as da
from dask import delayed, compute
import numpy as np
from tqdm import tqdm
import random
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides, load_sequences, save_features_to_parquet


def generate_kmers(k, alphabet='ACGT'):
    """Generate all possible k-mers from the given alphabet."""
    return [''.join(p) for p in product(alphabet, repeat=k)]

def find_spaced_words(sequence, pattern, pattern_indices, k):
    start = time.time()
    kmer_counts = defaultdict(int)  # Using defaultdict for faster updates
    all_kmers = generate_kmers(k)
    for i in range(len(sequence) - len(pattern) + 1):
    # Preallocate list to avoid list comprehension overhead
        spaced_word_chars = [None] * len(pattern_indices)
        for index, j in enumerate(pattern_indices):
            spaced_word_chars[index] = sequence[i + j]
        spaced_word = ''.join(spaced_word_chars)
        kmer_counts[spaced_word] += 1
    
    # Only include k-mers present in all_kmers to match original function's behavior
    feature_vector = np.array([kmer_counts[kmer] for kmer in all_kmers])
    return feature_vector

def generate_random_pattern(k, l):
    one_positions = random.sample(range(2, l + k + 1), k - 1)
    one_positions.append(1)
    one_positions.sort()
    
    # Create the pattern, filling in '0's where there isn't a '1'
    pattern = ''
    for i in range(1, max(one_positions) + 1):
        if i in one_positions:
            pattern += '1'
        else:
            pattern += '0'
    
    return pattern

def generate_spaced_word_features(pattern, sequence_df, k):
    pattern_indices = [i for i, char in enumerate(pattern) if char == '1']

    delayed_find_spaced_words = delayed(find_spaced_words)
    tasks = [delayed_find_spaced_words(sequence, pattern, pattern_indices, k) for sequence in sequence_df["Sequence"]]

    # Instantiate and register the ProgressBar
    progress_bar = ProgressBar()
    progress_bar.register()

    # Use dask.compute to execute tasks in parallel, now with a progress bar
    with progress_bar:
        feature_vectors = compute(*tasks, scheduler='processes')

    # Stack the resulting feature vectors
    stacked_feature_vectors = np.vstack(feature_vectors)
    return pd.DataFrame(stacked_feature_vectors)

def save_row(row, dataset, k):
    directory = f'results/011_spaced_patterns/{dataset}'
    csv_file_path = os.path.join(directory, f'pattern_weight_{k}_scores.csv')

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    file_exists = os.path.isfile(csv_file_path)
    mode = 'a' if file_exists else 'w'
    row.to_csv(csv_file_path, mode=mode, header=not file_exists, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process DNA sequences with Chaos Game Representation.')
    parser.add_argument('-d', '--Degenerate', choices=['replace', 'remove'], required=True, 
                        help='Specify the action on degenerate nucleotides: replace them or remove them entirely.')
    parser.add_argument('-k', '--Word_Length', type=int, choices=[5, 6, 7], required=True, 
                        help='Define the length of k-mers to be used in RTD calculations.')
    parser.add_argument('-v', '--Data', choices=['SARS-CoV-2', 'HIV'], required=True,
                        help='Specify the virus dataset.')
    parser.add_argument('-n', '--Num_Patterns', type=int, default=50, 
                        help='Specify the number of spaced patterns.')
    args = parser.parse_args()

    sequence_df = load_sequences(args)

    # Example usage
    k = args.Word_Length
    l = 30  # Length of the pattern
    
    best_accuracy = 0
    
    targets = sequence_df["Target"].tolist()
    train_labels = sequence_df["Train"].tolist()
    
    sequence_df.reset_index(drop=True, inplace=True)
    
    for i in range(args.Num_Patterns):
        pattern = generate_random_pattern(k, l)
    
        print(f"Generating Pattern: {i}")
        SWF_features = generate_spaced_word_features(pattern, sequence_df, k)
        SWF_features.reset_index(drop=True, inplace=True)

        X_train = SWF_features[sequence_df["Train"] == 0]
        y_train = sequence_df["Target"][sequence_df["Train"] == 0]
        
        X_val = SWF_features[sequence_df["Train"] == 1]
        y_val = sequence_df["Target"][sequence_df["Train"] == 1]
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        
        row_dict = {
            'pattern': pattern,        
            'accuracy': accuracy,
            'macro_f1': macro_f1
        }
                        
        row = pd.DataFrame([row_dict], index=[i])
        save_row(row, args.Data, k)
        
        if accuracy > best_accuracy:
            print(f'New Best Accuracy: {accuracy}')
            best_accuracy = accuracy
            save_features_to_parquet(SWF_features, args, targets, train_labels)
    
if __name__ == "__main__":
    main()