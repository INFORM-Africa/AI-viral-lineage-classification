import argparse
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
import re

def split_filename(filename):
    # Regular expression to match the pattern, allowing for 'remove' or 'replace'
    match = re.match(r'(\d+-spaced_(?:remove|replace)_)([01]+)', filename)
    if match:
        prefix = match.group(1)  # The prefix part
        binary_pattern = match.group(2)  # The binary pattern
        return prefix[:-1], binary_pattern
    else:
        return None, None

def load_data(file_name, pattern):
    """
    Load the data from a Parquet file, encode the target variable, and split the data into training and validation sets.

    Parameters:
    - file_name (str): Name of the file to load (without '.parquet' extension and path).

    Returns:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training labels.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation labels.
    """
    data_path = f'../../data/features/spaced/{file_name}_{pattern}.parquet'
    data = pd.read_parquet(data_path)
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Target"])

    X_train = data[data['Train'] == 0].drop(columns=["Target", "Train", "Label"])
    y_train = data[data['Train'] == 0]['Label']
    X_val = data[data['Train'] == 1].drop(columns=["Target", "Train", "Label"])
    y_val = data[data['Train'] == 1]['Label']

    return X_train, y_train, X_val, y_val

def evaluate_model(X_train, y_train, X_val, y_val, params, file_name, pattern):
    """
    Train a Random Forest model with the given parameters and evaluate it on the validation set.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training labels.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation labels.
    - params (dict): Parameters for the Random Forest model.

    Returns:
    - result (dict): Dictionary containing the parameters and evaluation metrics.
    """
    print(f"Training Random Forest: {params}")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    micro_f1 = f1_score(y_val, y_pred, average='micro')
    macro_f1 = f1_score(y_val, y_pred, average='macro')

    result = {
        'features': file_name,
        'pattern': pattern,
        'max_depth': params['max_depth'],
        'criterion': params['criterion'],
        'class_weight': params['class_weight'],
        'accuracy': accuracy,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }

    return result

def grid_search(X_train, y_train, X_val, y_val, param_grid, file_name, pattern):
    """
    Perform a grid search over the specified parameter grid for a Random Forest model and save the results.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training labels.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation labels.
    - param_grid (dict): Grid of parameters to search over.
    - file_name (str): Base name for the output CSV file to store the results.
    """
    results = []

    for params in ParameterGrid(param_grid):
        result = evaluate_model(X_train, y_train, X_val, y_val, params, file_name, pattern)
        results.append(result)
        print(f"Params: {params}, Accuracy: {result['accuracy']}, Micro F1: {result['micro_f1']}, Macro F1: {result['macro_f1']}")

    results_df = pd.DataFrame(results)
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Random Forest Grid Search with Command Line Arguments")
    parser.add_argument('file_name', type=str, help='Name of the file to process (without extension and path)')
    args = parser.parse_args()

    file_name, pattern = split_filename(args.file_name)
    X_train, y_train, X_val, y_val = load_data(file_name, pattern)

    param_grid = {
        'max_depth': [20, 30, 40, None],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }

    # Save the results to CSV
    results_df = grid_search(X_train, y_train, X_val, y_val, param_grid, file_name, pattern)
    csv_file_path = f'results/spaced_results.csv'
    file_exists = os.path.isfile(csv_file_path)
    mode = 'a' if file_exists else 'w'
    results_df.to_csv(csv_file_path, mode=mode, header=not file_exists, index=False)

if __name__ == "__main__":
    main()
