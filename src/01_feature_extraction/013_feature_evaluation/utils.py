import argparse
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

def load_data(dataset):
    """
    Load the data from a Parquet file and split the data into training and testing sets, while also retaining the original targets for the test set.

    Parameters:
    - file_name (str): Name of the file to load (without '.parquet' extension and path).

    Returns:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training labels.
    - X_test (DataFrame): Testing features.
    - y_test_orig (Series): Original testing labels (before encoding).
    - label_encoder (LabelEncoder): The label encoder used for the target variable.
    """
    data_path = f'../../data/features/{dataset}.parquet'
    data = pd.read_parquet(data_path)

    X_train = data[data['Train'] == 0].drop(columns=["Target", "Train"])
    y_train = data[data['Train'] == 0]['Target']
    X_val = data[data['Train'] == 1].drop(columns=["Target", "Train"])
    y_val = data[data['Train'] == 1]['Target']
    X_test = data[data['Train'] == 2].drop(columns=["Target", "Train"])
    y_test = data[data['Train'] == 2]['Target']

    return X_train, y_train, X_val, y_val, X_test, y_test

def predict_and_save(model, X_test, y_test, feature_desc, prediction_set, dataset, model_name):
    """
    Train a Random Forest model with the given parameters, predict on the testing set, inverse transform the predictions to original targets, and save the predictions.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training labels.
    - X_test (DataFrame): Testing features.
    - y_test_orig (Series): Original testing labels.
    - label_encoder (LabelEncoder): The label encoder used for the target variable.
    - params (dict): Parameters for the Random Forest model.
    - file_name (str): Base name for the CSV file to store the predictions.
    """
    
    print(f"Predicting {prediction_set}")
    predictions = model.predict(X_test)

    # File path for the predictions CSV
    predictions_file_path = f'results/013_final_model_predictions/{dataset}/{model_name}_{prediction_set}_predictions.csv'

    # Create the directory if it does not exist
    directory = os.path.dirname(predictions_file_path)
    os.makedirs(directory, exist_ok=True)

    if os.path.exists(predictions_file_path):
        predictions_df = pd.read_csv(predictions_file_path)
        # Ensure the new column does not overwrite existing ones
        if feature_desc in predictions_df.columns:
            raise ValueError(f"Column '{feature_desc}' already exists in the predictions file.")
    else:
        predictions_df = pd.DataFrame()
        predictions_df['True Target'] = y_test.values

    # Add the new predictions as a column
    predictions_df[feature_desc] = predictions  # Assuming 'predictions' is defined and has the correct values

    # Save the updated or new predictions DataFrame to CSV
    predictions_df.to_csv(predictions_file_path, index=False)