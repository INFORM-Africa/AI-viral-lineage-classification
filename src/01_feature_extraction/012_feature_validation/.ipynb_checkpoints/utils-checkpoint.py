import argparse
import pandas as pd
import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def load_data(dataset):
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
    data_path = f'../../data/features/{dataset}.parquet'
    data = pd.read_parquet(data_path)

    X_train = data[data['Train'] == 0].drop(columns=["Target", "Train"])
    y_train = data[data['Train'] == 0]['Target']
    X_val = data[data['Train'] == 1].drop(columns=["Target", "Train"])
    y_val = data[data['Train'] == 1]['Target']

    return X_train, y_train, X_val, y_val

def save_model(model, model_name, dataset, folder, save_name = None):
    directory = f"models/{folder}/{dataset}"  # Define the directory path
    
    if save_name:
        filename = f"{directory}/{model_name}_{save_name}.joblib"
    else:
        filename = f"{directory}/{model_name}.joblib"
    
    if model_name == "xgboost":
        filename = filename.replace("joblib", "json")
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)  # This will create all intermediate directories as well
    
    if model_name == "xgboost":
        model.save_model(filename)
    else:
        dump(model, filename)

def evaluate_model(X_train, y_train, X_val, y_val, params, dataset, feature_desc, model_name, best_accuracy):
    print(f"Training {model_name.capitalize()}: {params}")
    
    model = None  # Initialize model to None to handle unexpected model_name values
    
    if model_name == "random_forest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        model.fit(X_train, y_train)
    elif model_name == "xgboost":
        model = XGBClassifier(tree_method='hist', device = "cuda", early_stopping_rounds=10, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_name == "knn":
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')

    result = params.copy()
    result['feature_desc'] = feature_desc
    result['accuracy'] = accuracy
    result['macro_f1'] = macro_f1
    

    if accuracy > best_accuracy:
        save_model(model, model_name, dataset, "active")
        save_model(model, model_name, dataset, "history", save_name = feature_desc)
        print(f"New best model saved: {accuracy} accuracy")
        
    if model_name == 'xgboost' and hasattr(model, 'best_iteration'):
        result['n_estimators'] = model.best_iteration

    return result, accuracy

def save_results(results_df, dataset, model_name):
    directory = f'results/012_validation_scores/{dataset}'
    csv_file_path = os.path.join(directory, f'{model_name}.csv')

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    file_exists = os.path.isfile(csv_file_path)
    mode = 'a' if file_exists else 'w'
    results_df.to_csv(csv_file_path, mode=mode, header=not file_exists, index=False)

def grid_search(param_grid, dataset, feature_desc, model_name):
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
    X_train, y_train, X_val, y_val = load_data(dataset)
    
    if model_name == 'knn':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.fit_transform(X_val)
    
    results = []
    
    best_accuracy = 0

    for params in ParameterGrid(param_grid):
        result, accuracy = evaluate_model(X_train, y_train, X_val, y_val, params, dataset, feature_desc, model_name, best_accuracy)
        best_accuracy = max(accuracy, best_accuracy)
        results.append(result)
        print(f"Params: {params}, Accuracy: {result['accuracy']}, Macro F1: {result['macro_f1']}")

    results_df = pd.DataFrame(results)
    
    save_results(results_df, dataset, model_name)