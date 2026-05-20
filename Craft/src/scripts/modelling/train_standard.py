import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np
import argparse
import os

def train(args):

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

    # Use the low_resource argument instead of hardcoded flag
    if args.low_resource:
        # Combine X_train_split and y_train for filtering
        train_data = pd.DataFrame(X_train)
        train_data['Target'] = y_train.tolist()
        
        # Sample 4 records per class
        limited_train_data = train_data.groupby('Target').apply(
            lambda x: x.sample(n=min(4, len(x)))
        ).reset_index(drop=True)
        
        # Separate back into features and targets
        X_train = limited_train_data.drop(columns=['Target']).to_numpy()
        y_train = limited_train_data['Target'].to_numpy()

    X_train = X_train.to_numpy()

    # Create output directory
    model_dir = f'../models/{args.model_name}'
    os.makedirs(model_dir, exist_ok=True)

    # Train the first Random Forest classifier
    rf = RandomForestClassifier(criterion=args.criterion, class_weight=args.class_weight)
    rf.fit(X_train, y_train)

    # Use num_features argument instead of hardcoded value
    if args.num_features == -1:
        # Use all features
        X_train_top_features = X_train
        top_features_indices = np.arange(X_train.shape[1])
    else:
        feature_importances = rf.feature_importances_
        top_features_indices = np.argsort(feature_importances)[-args.num_features:]
        X_train_top_features = X_train[:, top_features_indices]

    rf_top_features = RandomForestClassifier(criterion=args.criterion, class_weight=args.class_weight)
    rf_top_features.fit(X_train_top_features, y_train)

    # Save feature set to disk
    array_file = f"{model_dir}/feature_set.npy"
    np.save(array_file, top_features_indices)

    # Save Random Forest model to disk
    model_file = f"{model_dir}/forest.joblib"
    dump(rf_top_features, model_file)
    
    print(f"Model and feature set saved to: {model_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run a Random Forest on training data")
    parser.add_argument('--dataset', type=str, required=True, help='Viral Dataset name.')
    parser.add_argument('--run_name', type=str, required=True, help='Name for this run, used to create the output folder name.')
    parser.add_argument('--model_name', type=str, required=True, help='Name for the model, used in the output folder name.')
    parser.add_argument('--criterion', choices=['gini', 'entropy'], default='entropy', help='Splitting Criterion')
    parser.add_argument('--class_weight', choices=['balanced', 'None'], default='balanced', help='Class Weight')
    parser.add_argument('--low_resource', type=int, choices=[0, 1], default=0, help='Low resource mode: 0 for full dataset, 1 for limited dataset (4 samples per class)')
    parser.add_argument('--num_features', type=int, default=1000, help='Number of top features to use for second model. Use -1 for all features.')
    
    # Parse arguments
    args = parser.parse_args()
    if args.class_weight == 'None':
        args.class_weight = None
    
    train(args)

if __name__ == "__main__":
    main()