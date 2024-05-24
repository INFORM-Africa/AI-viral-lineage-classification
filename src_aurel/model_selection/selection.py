import logging
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from classifiers.metrics.metrics import h_f1_score

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y)
    
    train_indices = []
    test_indices = []
    
    for class_value in classes:
        class_mask = (y == class_value)
        class_indices = np.where(class_mask)[0]
        np.random.shuffle(class_indices)
        
        n_test_class = int(test_size * len(class_indices))
        n_test_class = min(n_test_class, len(class_indices) - 1)
        
        test_indices_class = class_indices[:n_test_class]
        train_indices_class = class_indices[n_test_class:]
        
        test_indices.extend(test_indices_class)
        train_indices.extend(train_indices_class)
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def _stratified_k_fold_split(y, n_splits=5, random_state=42):
    np.random.seed(random_state)
    classes, class_counts = np.unique(y, return_counts=True)
    
    # Filter out classes with fewer instances than the number of folds
    valid_classes = classes[class_counts >= n_splits]
    valid_indices = np.isin(y, valid_classes)
    
    y_valid = y[valid_indices]
    
    n_classes_discarded = len(classes) - len(valid_classes)
    n_samples_discarded = len(y) - len(y_valid)
    n_samples_left = len(y_valid)
    n_classes_left = len(valid_classes)

    logging.info(f"{n_samples_discarded} samples from {n_classes_discarded} class discarded for having fewer instances than the number of folds.")
    logging.info(f"Building the folds with {n_samples_left} samples from {n_classes_left} classes.")
    
    # Initialize folds
    folds = [[] for _ in range(n_splits)]
    
    # Distribute instances of each class to ensure at least one instance per fold
    for class_value in valid_classes:
        class_mask = (y_valid == class_value)
        class_indices = np.where(class_mask)[0]
        np.random.shuffle(class_indices)
        
        # Ensure at least one instance of each class in each fold
        for fold in range(n_splits):
            folds[fold].append(class_indices[fold])
        
        # Distribute remaining instances
        remaining_indices = class_indices[n_splits:]
        fold_sizes = np.full(n_splits, len(remaining_indices) // n_splits)
        fold_sizes[:len(remaining_indices) % n_splits] += 1
        
        current = 0
        for fold, fold_size in enumerate(fold_sizes):
            folds[fold].extend(remaining_indices[current:current + fold_size])
            current += fold_size
    
    for fold in folds:
        assert set(np.unique(y_valid[fold])) == set(valid_classes), "Folds must contain all classes"
    
    return folds, y_valid


def cross_val_score(estimator, X, y, n_splits=5, random_state=42):    
    folds, y = _stratified_k_fold_split(y, n_splits, random_state)
    scores = []
    unique_targets = np.unique(y)
    target_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_targets))}
    
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = []
        for j in range(n_splits):
            if j != i:
                train_indices.extend(folds[j])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        y_train = np.array([target_mapping[label] for label in y_train])
        y_test = np.array([target_mapping[label] for label in y_test])
        
        estimator_clone = clone(estimator)
        estimator_clone.fit(X_train, y_train)
        y_pred = estimator_clone.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
    
    return np.array(scores)


def h_cross_val_score(estimator, X, y, n_splits=5, random_state=42):    
    folds = _stratified_k_fold_split(y, n_splits, random_state)
    scores = []
    
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [idx for fold in folds if fold != folds[i] for idx in fold]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        estimator_clone = clone(estimator)
        estimator_clone.fit(X_train, y_train)
        y_pred = estimator_clone.predict(X_test)
        score = h_f1_score(y_test, y_pred)
        scores.append(score)
    
    return np.array(scores)


