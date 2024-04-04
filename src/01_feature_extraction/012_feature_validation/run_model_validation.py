import argparse
import pandas as pd
import os
from utils import grid_search, save_results

def main():
    parser = argparse.ArgumentParser(description="Random Forest Grid Search")
    parser.add_argument('-v', '--Data', choices=['SARS-CoV-2', 'HIV'], required=True,
                        help='Specify the virus dataset.')
    parser.add_argument('-f', '--Feature_Description', type=str, required=True,
                        help='Describe the feature set.')
    args = parser.parse_args()
    
    
    model_name = 'random_forest'

    feature_desc = args.Feature_Description
    print(f'Running Validation for {feature_desc}...')

    param_grid = {
        'max_depth': [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }
    
    grid_search(param_grid, args.Data, feature_desc, model_name)
    
    model_name = 'xgboost'
    print(f'Running XGBoost Validation for {feature_desc}...')

    param_grid = {
        'max_depth': [1,2,3,4,5],
        'n_estimators': [200],
        'eval_metric': ['mlogloss']
    }

    # Save the results to CSV
    grid_search(param_grid, args.Data, feature_desc, model_name)
    
    model_name = 'knn'
    print(f'Running KNN Validation for {feature_desc}...')

    param_grid = {
    'n_neighbors': list(range(1, 11)),  # K values from 1 to 10
    'weights': ['uniform', 'distance']  # Weight functions
    }

    # Save the results to CSV
    grid_search(param_grid, args.Data, feature_desc, model_name)
    
if __name__ == "__main__":
    main()