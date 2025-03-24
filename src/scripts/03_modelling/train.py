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

def save_results_to_csv(results, filename):
    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame([results])
    
    # Check if the CSV file already exists
    if path.exists(filename):
        # Read the existing data
        existing_df = pd.read_csv(filename)
        # Append the new data
        updated_df = pd.concat((existing_df, results_df))
    else:
        # If the file doesn't exist, use the new DataFrame as the starting point
        updated_df = results_df
    
    # Save the updated DataFrame to CSV
    updated_df.to_csv(filename, index=False)
    print(f"Results saved/updated in {filename}")

def train(client, args):
    data_path = f'../../../data/features/{args.dataset}/{args.path}/training_set'
    X_train = pd.read_parquet(data_path, engine='pyarrow')
    y_train = X_train['Target']
    X_train = X_train.drop(columns='Target')

    #Sanity Check for Validation
    if args.test == 1:
        forest = CustomRandomForest(client, save_path=f'../../models/{args.dataset}/{args.path}', criterion=args.criterion, class_weight=args.class_weight, n_estimators=10)
    else:
        forest = CustomRandomForest(client, save_path=f'../../models/{args.dataset}/{args.path}', criterion=args.criterion, class_weight=args.class_weight, n_estimators=10)

    forest.fit(X_train, y_train)
    
    forest.save_to_disk()
    del X_train
    del y_train
    return forest
    
def validate(forest, args):
    data_path = f'../../../data/features/{args.dataset}/{args.path}/validation_set'
    X_val = pd.read_parquet(data_path, engine='pyarrow')
    y_val = X_val['Target']
    X_val = X_val.drop(columns='Target')

    # Assume X_test is already loaded or prepared
    y_pred = forest.predict(X_val)

    tree_descriptions = forest.describe_trees()

    #Tree information
    avg_no_leaves = np.mean(tree_descriptions['number_of_leaves'])
    avg_depth = np.mean(tree_descriptions['depth'])
    avg_unique_features = np.mean(tree_descriptions['unique_features_used'])
    total_features = tree_descriptions['total_features'][0]
    
    # Calculate F1 scores
    val_accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    mcc = matthews_corrcoef(y_val, y_pred)

    # Example of setting up your results as a dictionary
    results = {
        'feature': args.path,
        'class_weight': tree_descriptions['class_weight'][0],
        'criterion': tree_descriptions['criterion'][0],
        'max_depth': tree_descriptions['max_depth'][0],
        'avg_depth': np.mean(tree_descriptions['depth']),
        'avg_unique_features': np.mean(tree_descriptions['unique_features_used']),
        'total_features': tree_descriptions['total_features'][0],
        'val_accuracy': accuracy_score(y_val, y_pred),
        'macro_f1': f1_score(y_val, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_val, y_pred)
    }
    
    # Call the function to save or update the results in the CSV
    os.makedirs(f'../../results/{args.dataset}', exist_ok=True)
    save_results_to_csv(results, f'../../results/{args.dataset}/rapid_validation_results.csv')

def predict(forest, client, args):
    sets = ['training_set', 'validation_set', 'testing_set']

    for dataset in sets:
        data_path = f'../../../data/features/{args.dataset}/{args.path}/{dataset}'  # Corrected from {set} to {dataset}
        X_test = pd.read_parquet(data_path, engine='pyarrow')
        y_test = X_test['Target']
        X_test = X_test.drop(columns='Target')

        # Predict using the loaded model
        y_pred = forest.predict(X_test)

        # File path for storing the results
        results_file = f'../../results/{args.dataset}/{dataset}_predictions_{args.run_num}.csv'

        os.makedirs(f'../../results/{args.dataset}', exist_ok=True)

        # Check if the results file exists
        if path.exists(results_file):
            # File exists, load it and add a new column for the current predictions
            results_df = pd.read_csv(results_file)
            results_df[args.path] = y_pred
        else:
            # File does not exist, create a new DataFrame
            results_df = pd.DataFrame({
                'y_true': y_test,
                args.path: y_pred
            })

        # Save or update the results DataFrame to CSV
        results_df.to_csv(results_file, index=False)
        print(f"Results saved/updated in {results_file} for dataset {dataset}")

def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Run a Custom Random Forest on training data")
    parser.add_argument('--dataset', choices=['SARS-CoV-2', 'DenV', 'HIV', 'DenV_NEW'], required=True, help='Specify viral dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the training set directory')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers for the Dask cluster')
    parser.add_argument('--criterion', choices=['gini', 'entropy'], default='gini', help='Splitting Criterion')
    parser.add_argument('--class_weight', choices=['balanced', 'None'], default='None', help='Class Weight')
    parser.add_argument('--run_num', type=int, default=0, help='Run number')
    parser.add_argument('--test', type=int, default=0, help='Class Weight')
    
    # Assuming you have 'args.feature', 'end', and 'start' defined somewhere in your code
    output_file = 'timings.txt'  
    
    # Parse arguments
    args = parser.parse_args()
    if args.class_weight == 'None':
        args.class_weight = None
    
    # Create a Dask client with a single worker and thread
    cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=1, memory_limit='24GB')
    client = Client(cluster)
    
    forest = train(client, args)
    train_time = time.time()
    output_text = f'{args.path} training: {train_time-start} - ({args.n_workers} workers)\n'
    with open(output_file, 'a') as file:
        file.write(output_text)

    # if args.test == 1:
    #     predict(forest, client, args)
    #     predict_time = time.time()
    #     output_text = f'{args.path} prediction: {predict_time-train_time} - ({args.n_workers} workers)\n'
    #     with open(output_file, 'a') as file:
    #         file.write(output_text)
    # else:
    #     validate(forest, args)

if __name__ == "__main__":
    main()