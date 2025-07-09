import pandas as pd
import numpy as np
import dask
import argparse
import os
from dask.distributed import Client, LocalCluster
from lib.modelling.custom_random_forest import CustomRandomForest
import time

def load_feature_data(features_path):
    """
    Load feature data from the specified path.
    Feature data is stored as a single parquet file.
    """
    print(f"Loading feature data from: {features_path}")
    
    # Read feature data
    X_data = pd.read_parquet(features_path, engine='pyarrow')
    print(f"Loaded data shape: {X_data.shape}")
    
    return X_data

def predict(client, args):
    """
    Load model and make predictions on feature data.
    """
    # Load feature data
    features_path = f'../features/{args.dataset}_{args.run_name}'
    X_data = load_feature_data(features_path)
    
    # Separate Accession ID from feature data
    if 'Accession ID' not in X_data.columns:
        raise ValueError("'Accession ID' column not found in feature data")
    
    accession_ids = X_data['Accession ID'].copy()
    X_features = X_data.drop(columns=['Accession ID'])
    
    # Load the trained model
    model_path = f'../models/{args.model_name}/forest.pkl'
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    forest = CustomRandomForest.load_from_disk(model_path, client)
    
    # Make predictions
    print("Predicting...")
    start_time = time.time()
    predictions = forest.predict(X_features)
    end_time = time.time()
    
    print(f"Predictions completed in {end_time - start_time:.2f} seconds")
    
    # Create output dataframe
    results_df = pd.DataFrame({
        'Accession ID': accession_ids,
        'Lineage': predictions
    })
    
    # Create predictions directory if it doesn't exist
    predictions_dir = '../predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save results to CSV
    output_path = f'{predictions_dir}/{args.dataset}_{args.run_name}_{args.model_name}_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Make predictions using a trained Custom Random Forest model")
    parser.add_argument('--dataset', type=str, required=True, help='Viral Dataset name.')
    parser.add_argument('--run_name', type=str, required=True, help='Name for this run, used to locate the model and features.')
    parser.add_argument('--model_name', type=str, required=True, help='Name for the model, used in the model folder name.')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers for the Dask cluster')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create a Dask client
    cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=1, memory_limit='24GB')
    client = Client(cluster)
    print(f"Created Dask client with {args.n_workers} workers")
    
    try:
        # Run predictions
        predict(client, args)
        print("Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise
    finally:
        # Clean up
        client.close()
        cluster.close()

if __name__ == "__main__":
    main() 