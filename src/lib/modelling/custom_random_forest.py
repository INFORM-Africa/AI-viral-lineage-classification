from dask.distributed import Client, LocalCluster
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
import os
import dask
import numpy as np
import joblib
import pandas as pd
import gc

class CustomRandomForest:
    def __init__(self, client, save_path,  n_estimators=100, max_depth=None, max_samples=None, fair_split=False, criterion='gini', class_weight=None):
        self.client = client
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.class_weight = class_weight
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.fair_split = fair_split
        self.tree_paths = []
        self.tree_stats = []
        self.save_path = save_path

    def fit(self, X, y):
        # Scatter data to the worker
        X_future, y_future = self.client.scatter([X, y], broadcast=True)

        os.makedirs(self.save_path, exist_ok=True)

        delayed_tasks = []
        for i in range(self.n_estimators):
            tree_path = f"{self.save_path}/tree_{i}.joblib"
            self.tree_paths.append(tree_path)

            # Create a Dask delayed task for each tree, including gathering stats
            task = dask.delayed(self.fit_save_tree)(X_future, y_future, tree_path, self.class_weight, self.criterion, self.max_depth, self.max_samples, self.fair_split)
            delayed_tasks.append(task)

        # Compute all delayed tasks and capture tree stats
        self.tree_stats = dask.compute(*delayed_tasks)

    @staticmethod
    def fit_save_tree(X, y, path, class_weight, criterion, max_depth, max_samples, fair_split):
        if fair_split:
            # Sample at least one random observation from each class
            unique_classes = y.unique()
            indices = []
            for cls in unique_classes:
                class_indices = y[y == cls].index
                indices.append(np.random.choice(class_indices, 1, replace=False))
            indices = np.squeeze(np.array(indices))
            
            if max_samples is not None:
                # Calculate remaining sample size
                sample_size = int(len(X) * max_samples) - len(unique_classes)
                remaining_indices = np.random.choice(X.index.difference(indices), sample_size, replace=False)
                indices = np.concatenate((indices, remaining_indices))
            else:
                # Add remaining indices with replacement for bootstrap
                remaining_indices = np.random.choice(X.index.difference(indices), len(X) - len(unique_classes), replace=True)
                indices = np.concatenate((indices, remaining_indices))
        else:
            if max_samples is None:
                # Bootstrap sample
                indices = np.random.choice(len(X), len(X), replace=True)
            else:
                # Sample without replacement
                sample_size = int(len(X) * max_samples)
                indices = np.random.choice(len(X), sample_size, replace=False)

        X_sample, y_sample = X.iloc[indices], y.iloc[indices]

        # Fit tree
        tree = DecisionTreeClassifier(max_depth=max_depth, max_features="sqrt", criterion=criterion, class_weight = class_weight)
        tree.fit(X_sample, y_sample)

        # Gather tree stats
        tree_stats = {
            'path': path,
            'class_weight': class_weight,
            'criterion': criterion,
            'max_depth': max_depth,
            'number_of_leaves': tree.get_n_leaves(),
            'depth': tree.get_depth(),
            'total_nodes': tree.tree_.node_count,
            'root_impurity': tree.tree_.impurity[0],
            'average_impurity_decrease': np.average(tree.tree_.impurity, weights=tree.tree_.n_node_samples),
            'feature_importances': tree.feature_importances_,
            'unique_features_used': len(np.unique(tree.tree_.feature[tree.tree_.feature >= 0])),
            'total_features': X_sample.shape[1]
        }

        # Save tree with compression
        joblib.dump(tree, path, compress=6)
        del tree  # Optionally clear memory
        gc.collect()

        return tree_stats

    def predict(self, X):
        X_future = self.client.scatter(X, broadcast=True)

        # Create Dask delayed tasks for each tree prediction
        delayed_predictions = [dask.delayed(self.load_and_predict)(tree_path, X_future) for tree_path in self.tree_paths]

        # Gather predictions from all trees
        predictions = dask.compute(*delayed_predictions)

        # Convert list of predictions to an array
        predictions = np.array(predictions)

        # Majority vote across all predictions
        if predictions.shape[0] > 1:
            # Use mode to find the most common class label among the predictions
            final_prediction, _ = mode(predictions, axis=0)
        else:
            final_prediction = predictions[0]

        return final_prediction

    @staticmethod
    def load_and_predict(path, X):
        # Load the trained tree from disk
        tree = joblib.load(path)
        prediction = tree.predict(X)
        del tree  # Optionally clear memory
        return prediction

    def describe_trees(self):
        # Create a DataFrame from the list of tree statistics
        df_trees = pd.DataFrame(self.tree_stats)
        return df_trees

    def set_trees(self, tree_paths):
        self.tree_paths = tree_paths

    def save_to_disk(self):
        # Save the entire model metadata to a file
        model_metadata = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'tree_paths': self.tree_paths,
            'tree_stats': self.tree_stats
        }
        joblib.dump(model_metadata, f'{self.save_path}/forest.pkl')

    @classmethod
    def load_from_disk(cls, path, client):
        # Load model metadata from disk
        model_metadata = joblib.load(path)
        
        # Extract the save_path from the path
        save_path = os.path.dirname(path)
        
        # Create a new instance of CustomRandomForest
        model = cls(client, save_path, n_estimators=model_metadata['n_estimators'], max_depth=model_metadata['max_depth'])
        model.tree_paths = model_metadata['tree_paths']
        model.tree_stats = model_metadata['tree_stats']

        return model