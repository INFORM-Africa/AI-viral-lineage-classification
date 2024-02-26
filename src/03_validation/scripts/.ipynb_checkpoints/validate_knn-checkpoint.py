import argparse
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def random_undersample(data_df, max_samples_per_class=3, random_state=42):

    undersampled_data = []

    for class_value, group in data_df.groupby('Label'):
        if len(group) > max_samples_per_class:
            undersampled_group = group.sample(n=max_samples_per_class, random_state=random_state)
        else:
            undersampled_group = group
        undersampled_data.append(undersampled_group)

    undersampled_data_df = pd.concat(undersampled_data)
    undersampled_data_df = undersampled_data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return undersampled_data_df

def load_data(file_name):
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
    data_path = f'../../data/features/{file_name}.parquet'
    data = pd.read_parquet(data_path)
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Target"])
    
    X_train = data[data['Train'] == 0].drop(columns=["Target"])
    # X_train = random_undersample(X_train)
    y_train = X_train[X_train['Train'] == 0]['Label']
    X_train = X_train.drop(columns=['Train', 'Label'])
    X_val = data[data['Train'] == 1].drop(columns=["Target", "Train", "Label"])
    y_val = data[data['Train'] == 1]['Label']

    return X_train, y_train, X_val, y_val

def evaluate_model(X_train_pca, y_train, X_val_pca, y_val, k, file_name):
    """
    Train a KNN model with the given k value and evaluate it on the PCA-transformed validation set.

    Parameters:
    - X_train_pca (ndarray): PCA-transformed training features.
    - y_train (Series): Training labels.
    - X_val_pca (ndarray): PCA-transformed validation features.
    - y_val (Series): Validation labels.
    - k (int): The number of neighbors to use for k-nearest neighbors.

    Returns:
    - result (dict): Dictionary containing the k value and evaluation metrics.
    """
    print(f"Training KNN with k={k}")
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_val_pca)

    accuracy = accuracy_score(y_val, y_pred)
    micro_f1 = f1_score(y_val, y_pred, average='micro')
    macro_f1 = f1_score(y_val, y_pred, average='macro')

    result = {
        'features': file_name,
        'k': k,
        'accuracy': accuracy,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }

    return result

def knn_grid_search(X_train, y_train, X_val, y_val, k_values, file_name):
    """
    Apply PCA once, then perform a grid search over the specified k values for a KNN model and save the results.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training labels.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation labels.
    - k_values (list of int): List of k values to search over.
    - file_name (str): Base name for the output file to store the results.
    """

#     pca_initial = PCA(n_components=min(1000, X_train.shape[1]), svd_solver='randomized')
#     X_train_pca_initial = pca_initial.fit_transform(X_train)
#     X_val_pca_initial = pca_initial.transform(X_val)
#     cumulative_variance = pca_initial.explained_variance_ratio_.cumsum()
#     # Find the number of components for 95% variance
#     n_components_95 = (cumulative_variance < 0.95).sum() + 1

#     print(f"Number of components to retain 95% variance: {n_components_95}")
        
#     X_train_pca = X_train_pca_initial[:, :n_components_95]
#     X_val_pca = X_val_pca_initial[:, :n_components_95]

    results = []

    for k in k_values:
        result = evaluate_model(X_train, y_train, X_val, y_val, k, file_name)
        results.append(result)
        print(f"k: {k}, Accuracy: {result['accuracy']}, Micro F1: {result['micro_f1']}, Macro F1: {result['macro_f1']}")

    results_df = pd.DataFrame(results)
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Random Forest Grid Search with Command Line Arguments")
    parser.add_argument('file_name', type=str, help='Name of the file to process (without extension and path)')
    args = parser.parse_args()

    file_name = args.file_name
    X_train, y_train, X_val, y_val = load_data(file_name)

    # Save the results to CSV
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results_df = knn_grid_search(X_train, y_train, X_val, y_val, k_values, file_name)
    csv_file_path = f'results/knn_results.csv'
    file_exists = os.path.isfile(csv_file_path)
    mode = 'a' if file_exists else 'w'
    results_df.to_csv(csv_file_path, mode=mode, header=not file_exists, index=False)

if __name__ == "__main__":
    main()
