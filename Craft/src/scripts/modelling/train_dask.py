from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pandas as pd
import numpy as np
import dask
import argparse
import os
from dask.distributed import Client, LocalCluster
from lib.modelling.custom_random_forest import CustomRandomForest
from os import path
import gc
import time

def train(client, args):
    # Read feature data
    data_path = f'../features/{args.dataset}_{args.run_name}'
    X_train = pd.read_parquet(data_path, engine='pyarrow')
    
    # Read target data from separate CSV file
    targets_path = f'../data/{args.dataset}_targets.csv'
    targets_df = pd.read_csv(targets_path)
    
    # Merge feature data with target data using Accession ID
    X_train = X_train.merge(targets_df, on='Accession ID', how='inner')
    
    # Extract target column and drop both Accession ID and Target columns
    y_train = X_train['Target']
    X_train = X_train.drop(columns=['Accession ID', 'Target'])

    forest = CustomRandomForest(client, save_path=f'../models/{args.model_name}', criterion=args.criterion, class_weight=args.class_weight, n_estimators=10)

    forest.fit(X_train, y_train)
    
    forest.save_to_disk()
    del X_train
    del y_train
    return forest

def main():
    parser = argparse.ArgumentParser(description="Run a Custom Random Forest on training data")
    parser.add_argument('--dataset', type=str, required=True, help='Viral Dataset name.')
    parser.add_argument('--run_name', type=str, required=True, help='Name for this run, used to create the output folder name.')
    parser.add_argument('--model_name', type=str, required=True, help='Name for the model, used in the output folder name.')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers for the Dask cluster')
    parser.add_argument('--criterion', choices=['gini', 'entropy'], default='gini', help='Splitting Criterion')
    parser.add_argument('--class_weight', choices=['balanced', 'None'], default='None', help='Class Weight')
    
    # Parse arguments
    args = parser.parse_args()
    if args.class_weight == 'None':
        args.class_weight = None
    
    # Create a Dask client with a single worker and thread
    cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=1, memory_limit='24GB')
    client = Client(cluster)
    
    train(client, args)

if __name__ == "__main__":
    main()