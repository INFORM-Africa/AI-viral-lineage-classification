import argparse
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import re

def load_data(file_name, dataset):
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
    data_path = f'../../../data/features/{dataset}/{file_name}.parquet'
    data = pd.read_parquet(data_path)
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Target"])

    X_train = data[data['Train'] == 0].drop(columns=["Target", "Train", "Label"])
    y_train = data[data['Train'] == 0]['Label']
    y_train_orig = data[data['Train'] == 0]['Target']
    X_val = data[data['Train'] == 1].drop(columns=["Target", "Train", "Label"])
    y_val_orig = data[data['Train'] == 1]['Target']
    X_test = data[data['Train'] == 2].drop(columns=["Target", "Train", "Label"])
    y_test_orig = data[data['Train'] == 2]['Target']

    return X_train, y_train, y_train_orig, X_val, y_val_orig, X_test, y_test_orig, label_encoder

def predict_and_save(model, X_test, y_test_orig, label_encoder, file_name, prediction_set, dataset):
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
    predictions_encoded = model.predict(X_test)
    predictions = label_encoder.inverse_transform(predictions_encoded)

    # File path for the predictions CSV
    predictions_file_path = f'predictions/{dataset}/xgboost_{prediction_set}.csv'

    # Create the directory if it does not exist
    directory = os.path.dirname(predictions_file_path)
    os.makedirs(directory, exist_ok=True)

    if os.path.exists(predictions_file_path):
        predictions_df = pd.read_csv(predictions_file_path)
        # Ensure the new column does not overwrite existing ones
        if file_name in predictions_df.columns:
            raise ValueError(f"Column '{file_name}' already exists in the predictions file.")
    else:
        predictions_df = pd.DataFrame()
        predictions_df['Original_Targets'] = y_test_orig.values  # Assuming y_test_orig is defined and has the correct values

    # Add the new predictions as a column
    predictions_df[file_name] = predictions  # Assuming 'predictions' is defined and has the correct values

    # Save the updated or new predictions DataFrame to CSV
    predictions_df.to_csv(predictions_file_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="XGBoost Model Prediction and CSV Update")
    parser.add_argument('file_name', type=str, help='Name of the file to process (without extension and path)')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of gradient boosted trees. Equivalent to number of boosting rounds.')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum tree depth for base learners.')
    parser.add_argument('-v', '--Data', choices=['SARS', 'HIV'], required=True,
                        help='Specify the virus dataset.')
    args = parser.parse_args()

    file_name = args.file_name
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth
    }

    X_train, y_train, y_train_orig, X_val, y_val_orig, X_test, y_test_orig, label_encoder = load_data(file_name, args.Data)
    
    print("Training")
    model = XGBClassifier(random_state=42, tree_method='hist', device = "cuda", use_label_encoder=False, eval_metric='mlogloss', **params)
    model.fit(X_train, y_train)
    
    predict_and_save(model, X_train, y_train_orig, label_encoder, file_name, "Training", args.Data)
    predict_and_save(model, X_val, y_val_orig, label_encoder, file_name, "Validation", args.Data)
    predict_and_save(model, X_test, y_test_orig, label_encoder, file_name, "Testing", args.Data)
    
    print(f"Predictions for '{file_name}' with original targets saved.")

if __name__ == "__main__":
    main()