import os
import dask
import gc
import argparse
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask import delayed, compute
from dask.distributed import Client, LocalCluster, get_worker
import psutil
import threading
import time

from lib.features.chaos_game import chaos_game_representation
from lib.features.spaced_word_frequency import spaced_word_representation, generate_random_pattern
from lib.features.genomic_signal_processing import genomic_signal_representation, pearson_correlation_dissimilarity
from lib.features.kmer import kmer_representation
from lib.features.mash import mash_sketch_representation, mash_distance
from lib.features.return_time_distribution import return_time_representation
from lib.features.utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides
import sys

@delayed
def generate_spaced_word_representation(partition, pattern, args):
    return spaced_word_representation(partition, pattern = pattern, k=args.k)
    
@delayed
def generate_chaos_game_representation(partition, args):
    return chaos_game_representation(partition, resolution=args.res)

@delayed
def generate_kmer_representation(partition, args):
    return kmer_representation(partition, k=args.k)

@delayed
def generate_genomic_signal_representation(partition, median_length, args):
    return genomic_signal_representation(partition, median_length, mapping=args.map)

@delayed
def generate_mash_sketch_representation(partition, args):
    return mash_sketch_representation(partition, k=args.k, sketch_size=args.size)

@delayed
def generate_return_time_representation(partition, args):
    return return_time_representation(partition, k=args.k)

@delayed
def handle_degenerates(partition, args):
    if args.deg == 'remove':
        return remove_degenerate_nucleotides(partition)
    elif args.deg == 'replace':
        return replace_degenerate_nucleotides(partition)
    else:
        raise ValueError("Invalid option for degenerate nucleotide handling")

@delayed
def save_features_to_disk(features_representation, partition, index, output_path):
    df = pd.DataFrame(features_representation, columns=[f'feature {j+1}' for j in range(features_representation.shape[1])])
    df['Target'] = partition['Target'].tolist()
    partition_output_path = os.path.join(output_path, f'fp_{index}.parquet')
    df.to_parquet(partition_output_path, engine='pyarrow', index=False)
    del df
    gc.collect()

@delayed
def calculate_distance_matrix(features, distance_transform, args):
    return mash_distance(features, distance_transform, args.k)

@delayed
def calculate_correlation_matrix(features, distance_transform):
    return pearson_correlation_dissimilarity(features, distance_transform)

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process and analyze genomic sequences based on various feature extraction methods.')

    # Feature type
    parser.add_argument(
        '--dataset',
        choices=['SARS-CoV-2', 'DenV', 'HIV', 'DenV_NEW'],
        required=True,
        help='Viral Dataset. Choices include SARS-CoV-2, DenV and HIV.'
    )
    
    parser.add_argument(
        '--feature',
        type=str.upper,
        choices=['FCGR', 'ACS', 'KMER', 'SWF', 'GSP', 'MASH', 'RTD'],
        required=True,
        help='Type of feature extraction to perform. Choices include FCGR, ACS, kmer, SWF, GSP, Mash, and RTD.'
    )

    # Degenerate nucleotides handling
    parser.add_argument(
        '--deg',
        choices=['remove', 'replace'],
        required=True,
        help='Specifies how degenerate nucleotides should be handled: removed or replaced randomly.'
    )

    # Word length (k)
    parser.add_argument(
        '--k',
        type=int,
        help='Specifies the word length for kmer, Mash, SWF, and RTD feature extraction.'
    )

    # Image resolution for FCGR
    parser.add_argument(
        '--res',
        type=int,
        choices=[32, 64, 128, 256],
        help='Specifies the image resolution for FCGR feature extraction.'
    )

    # Sketch size for Mash
    parser.add_argument(
        '--size',
        type=int,
        choices=[1000, 2000],
        help='Specifies the sketch size for Mash feature extraction.'
    )

    # Numeric mapping for GSP
    parser.add_argument(
        '--map',
        choices=['real', 'eiip', 'justa', 'pp'],
        help='Specifies the form of numeric mapping for GSP feature extraction.'
    )

    # Spaced pattern for SWF
    parser.add_argument(
        '--pattern',
        type=str,
        help='Specifies the spaced pattern for SWF feature extraction.'
    )

    return parser

def validate_args(args, parser):
    """Validate the inter-dependency of command line arguments based on the feature type."""
    feature_needs = {
        'FCGR': ('res',),
        'KMER': ('k',),
        'SWF': ('k'),
        'GSP': ('map',),
        'MASH': ('k', 'size'),
        'RTD': ('k',)
    }

    needed = feature_needs.get(args.feature, ())
    missing = [arg for arg in needed if getattr(args, arg) is None]

    if missing:
        parser.error(f"Feature {args.feature.upper()} requires the following missing arguments: {', '.join(missing)}")

def monitor_memory_usage():
    """Monitor and record maximum memory usage."""
    process = psutil.Process()
    max_memory = 0
    
    while True:
        try:
            current_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            max_memory = max(max_memory, current_memory)
            time.sleep(1)  # Check every second
        except:
            break
    
    return max_memory

def main():
    start = time.time()
    parser = create_arg_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    
    # Start memory monitoring in a separate thread
    memory_stop_flag = threading.Event()
    max_memory = [0]  # Use list to store value that can be modified by thread
    
    def memory_monitor_thread():
        process = psutil.Process()
        while not memory_stop_flag.is_set():
            try:
                current_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
                max_memory[0] = max(max_memory[0], current_memory)
                time.sleep(1)  # Check every second
            except:
                break
    
    memory_thread = threading.Thread(target=memory_monitor_thread)
    memory_thread.start()
    
    # Initialize Dask Client
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        memory_limit='4GB'
    )
    
    client = Client(cluster)
    print(f"Number of workers: {len(client.cluster.workers)}")
    print(f"Threads per worker: {client.cluster.workers[0].nthreads}")

    # Load dataset and set parameters
    data_path = f'../../../data/sequences/{args.dataset}'
    base_output_path = f'../../../data/features/{args.dataset}'
    sets = ['training_set']

    feature_to_function = {
        'FCGR': generate_chaos_game_representation,
        'KMER': generate_kmer_representation,
        'GSP': generate_genomic_signal_representation,
        'MASH': generate_mash_sketch_representation,
        'RTD': generate_return_time_representation,
        'SWF': generate_spaced_word_representation,
    }

    path_mapping = {
        'FCGR': f"{args.res}",
        'RTD': f"{args.k}",
        'KMER': f"{args.k}",
        'MASH': f"{args.k}/{args.size}",
        'GSP': f"{args.map}",
        'SWF': f"{args.k}"
    }
    
    extract_features = feature_to_function[args.feature]
    params_path = path_mapping[args.feature]

    median_length = 0

    if args.feature == 'MASH':
        distance_sample = pd.read_parquet('distance_sample.parquet', engine='pyarrow')
        # distance_sample = pd.read_parquet(f"{data_path}/training_set", engine='pyarrow')
        distance_transform = extract_features(distance_sample, args).compute()
        distance_transform_future = client.scatter(distance_transform, broadcast=True)
        # distance_sample = pd.read_parquet('distance_sample.parquet', engine='pyarrow')
        # distance_transform = extract_features(distance_sample, args).compute()
        # distance_transform_future = client.scatter(distance_transform, broadcast=True)

    if args.feature == 'GSP':
        # distance_sample = pd.read_parquet(f"{data_path}/training_set", engine='pyarrow'
        distance_sample = pd.read_parquet('distance_sample.parquet', engine='pyarrow')
        sequence_lengths = distance_sample['Sequence'].apply(len)
        median_length = int(sequence_lengths.median())
        distance_transform = extract_features(distance_sample, median_length, args).compute()
        distance_transform_future = client.scatter(distance_transform, broadcast=True)

    l = 30  # Length of the pattern
    if args.feature == 'SWF': 
        pattern = generate_random_pattern(args.k, l)
        # pattern = '101000000100000101101'

    for set in sets:
        sequence_df = dd.read_parquet(f"{data_path}/{set}", engine='pyarrow')
    
        partitions = sequence_df.to_delayed()

        output_path = f"{base_output_path}/{args.feature}/{args.deg}/{params_path}/{set}"
        os.makedirs(output_path, exist_ok=True)
    
        tasks = []
        for index, partition in enumerate(partitions):
            partition = handle_degenerates(partition, args)
            if args.feature == 'GSP':
                features = extract_features(partition, median_length, args)
            elif args.feature == 'SWF':
                features = extract_features(partition, pattern, args)
            else:
                features = extract_features(partition, args)
            if args.feature == 'MASH':
                features = calculate_distance_matrix(features, distance_transform_future, args)
            if args.feature == 'GSP':
                features = calculate_correlation_matrix(features, distance_transform_future)
            save_task = save_features_to_disk(features, partition, index, output_path)
            tasks.append(save_task)
    
        compute(*tasks)
    
    end = time.time()
    
    # Stop memory monitoring
    memory_stop_flag.set()
    memory_thread.join()
    
    # Record both time and memory usage
    total_time = end - start
    output_text = f'Time (s): {total_time}\nMax Memory (MB): {max_memory[0]}\n'

    # Define the name of the output file
    output_file = 'performance_metrics.txt'

    # Open the file in append mode and write the output text
    with open(output_file, 'a') as file:
        file.write(f"\n=== {args.feature} ===\n")
        file.write(output_text)

    print(f'Performance metrics written to {output_file}')
    
    # Close the client
    client.close()

if __name__ == "__main__":
    main()