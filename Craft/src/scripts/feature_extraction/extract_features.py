import os
import dask
import gc
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask import delayed, compute
from dask.distributed import Client, LocalCluster, get_worker
import time

from lib.feature_extraction.chaos_game import chaos_game_representation
from lib.feature_extraction.spaced_word_frequency import spaced_word_representation, generate_random_pattern
from lib.feature_extraction.genomic_signal_processing import genomic_signal_representation, pearson_correlation_dissimilarity
from lib.feature_extraction.kmer import kmer_representation
from lib.feature_extraction.mash import mash_sketch_representation, mash_distance
from lib.feature_extraction.return_time_distribution import return_time_representation
from lib.feature_extraction.utils import replace_degenerate_nucleotides, remove_degenerate_nucleotides
from parser import create_arg_parser, validate_args
import sys

@delayed
def generate_spaced_word_representation(partition, pattern, args):
    return spaced_word_representation(partition, pattern = pattern, k=args.k)
    
@delayed
def generate_chaos_game_representation(partition, args):
    return chaos_game_representation(partition, resolution=args.res, mode=args.mode)

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
    df['Accession ID'] = partition['Accession ID'].tolist()
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


def main():
    start = time.time()
    parser = create_arg_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    
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
    data_path = f'../data/{args.dataset}'
    base_output_path = f'../features/{args.dataset}_{args.run_name}'

    feature_to_function = {
        'FCGR': generate_chaos_game_representation,
        'KMER': generate_kmer_representation,
        'GSP': generate_genomic_signal_representation,
        'MASH': generate_mash_sketch_representation,
        'RTD': generate_return_time_representation,
        'SWF': generate_spaced_word_representation,
    }

    # Remove the path_mapping since we're simplifying the output structure
    extract_features = feature_to_function[args.feature]

    if args.feature == 'MASH':
        # distance_sample = pd.read_parquet('distance_sample.parquet', engine='pyarrow')
        distance_sample = pd.read_parquet(f"{data_path}.parquet", engine='pyarrow')
        distance_transform = extract_features(distance_sample, args).compute()
        distance_transform_future = client.scatter(distance_transform, broadcast=True)

    if args.feature == 'GSP':
        distance_sample = pd.read_parquet(f"{data_path}.parquet", engine='pyarrow')
        # distance_sample = pd.read_parquet('distance_sample.parquet', engine='pyarrow')
        sequence_lengths = distance_sample['Sequence'].apply(len)
        median_length = int(sequence_lengths.median())
        distance_transform = extract_features(distance_sample, median_length, args).compute()
        distance_transform_future = client.scatter(distance_transform, broadcast=True)

    l = 30  # Length of the pattern
    if args.feature == 'SWF': 
        pattern = generate_random_pattern(args.k, l)

    sequence_df = dd.read_parquet(f"{data_path}.parquet", engine='pyarrow')

    partitions = sequence_df.to_delayed()

    output_path = base_output_path
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
    
    # Record execution time
    total_time = end - start
    print(f'Total execution time: {total_time:.2f} seconds')
    
    # Close the client
    client.close()

if __name__ == "__main__":
    main()