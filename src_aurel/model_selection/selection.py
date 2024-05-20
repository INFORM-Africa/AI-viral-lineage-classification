import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

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


def stratified_k_fold_split(y, n_splits=5, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y)
    folds = [[] for _ in range(n_splits)]
    
    for class_value in classes:
        class_mask = (y == class_value)
        class_indices = np.where(class_mask)[0]
        np.random.shuffle(class_indices)
        
        fold_sizes = np.full(n_splits, len(class_indices) // n_splits)
        fold_sizes[:len(class_indices) % n_splits] += 1
        
        current = 0
        for fold, fold_size in enumerate(fold_sizes):
            folds[fold].extend(class_indices[current:current + fold_size])
            current += fold_size
    
    return folds


def cross_val_score(estimator, X, y, n_splits=5, random_state=42):
    folds = stratified_k_fold_split(y, n_splits, random_state)
    scores = []
    
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [idx for fold in folds if fold != folds[i] for idx in fold]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        estimator_clone = clone(estimator)
        estimator_clone.fit(X_train, y_train)
        y_pred = estimator_clone.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
    
    return np.array(scores)

# # Example usage
# from sklearn.ensemble import RandomForestClassifier

# X = np.random.rand(100, 4)  # 100 samples with 4 features each
# y = np.random.randint(0, 10, 100)  # 100 target values with 10 classes

# estimator = RandomForestClassifier(random_state=42)
# scores = cross_val_score(estimator, X, y, cv=5, random_state=42)

# print("Cross-validation scores:", scores)
# print("Mean accuracy:", scores.mean())

