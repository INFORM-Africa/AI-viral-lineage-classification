import pandas as pd
import numpy as np
import argparse
import os
import joblib
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

def predict(args):
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
    model_path = f'../models/{args.model_name}/forest.joblib'
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load the feature set indices
    feature_set_path = f'../models/{args.model_name}/feature_set.npy'
    print(f"Loading feature set from: {feature_set_path}")
    
    if not os.path.exists(feature_set_path):
        raise FileNotFoundError(f"Feature set not found at {feature_set_path}")
    
    # Load model and feature set
    rf_model = joblib.load(model_path)
    top_features_indices = np.load(feature_set_path)
    
    print(f"Loaded model with {rf_model.n_estimators} trees")
    print(f"Feature set contains {len(top_features_indices)} features")
    
    # Convert features to numpy array and select top features
    X_features_array = X_features.to_numpy()
    X_features_selected = X_features_array[:, top_features_indices]
    
    print(f"Selected feature data shape: {X_features_selected.shape}")
    
    # Make predictions
    print("Predicting...")
    start_time = time.time()
    predictions = rf_model.predict(X_features_selected)
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
    parser = argparse.ArgumentParser(description="Make predictions using a trained standard Random Forest model")
    parser.add_argument('--dataset', type=str, required=True, help='Viral Dataset name.')
    parser.add_argument('--run_name', type=str, required=True, help='Name for this run, used to locate the model and features.')
    parser.add_argument('--model_name', type=str, required=True, help='Name for the model, used in the model folder name.')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Run predictions
        predict(args)
        print("Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    main()

