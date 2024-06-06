import logging, utils
import numpy as np
from sklearn.base import clone
from classifiers import metrics as hmetrics
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocessing.read_data import get_hierarchy, normalize_hierarchies

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

def train_test_split_score(estimator, X, y, test_size=0.2, random_state=42):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= 2]

    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    y = LabelEncoder().fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
    return score

def h_train_test_split_score(estimator, X, y, test_size=0.2, random_state=42):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= 2]

    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    y = list(map(get_hierarchy, y))
    y = normalize_hierarchies(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    score = hmetrics.h_f1_score(y_test, y_pred)
    
    return score


def train_test_split_predict(estimator, X, y, path, test_size=0.2, random_state=42):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= 2]

    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    y = LabelEncoder().fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    utils.dump_parquet(data=y_pred, path=path)

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy}")
    mcc = matthews_corrcoef(y_test, y_pred)
    logging.info(f"MCC: {mcc}")
    report = classification_report(y_test, y_pred)
    logging.info(f"\n{report}")
    
    return accuracy, mcc, report

def h_train_test_split_predict(estimator, X, y, test_size=0.2, random_state=42):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= 2]

    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    y = list(map(get_hierarchy, y))
    y = normalize_hierarchies(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    report = hmetrics.h_classification_report(y_test, y_pred)
    
    return report


def cross_val_score(estimator, X, y, n_splits=5, random_state=42):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= n_splits]

    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    y = LabelEncoder().fit_transform(y)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scores = []
    i = 0

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        estimator_clone = clone(estimator)
        estimator_clone.fit(X_train, y_train)
        y_pred = estimator_clone.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        logging.info(f"Fold {i + 1} - Accuracy: {score}")
        i += 1
        
        scores.append(score)
    
    return np.array(scores)


# def h_cross_val_score(estimator, X, y, n_splits=5, random_state=42):
#     unique_classes, class_counts = np.unique(y, return_counts=True)
#     valid_classes = unique_classes[class_counts >= n_splits]

#     mask = np.isin(y, valid_classes)
#     X = X[mask]
#     y = y[mask]
#     y = list(map(get_hierarchy, y))
#     y = normalize_hierarchies(y)
    
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
#     scores = []
#     i = 0

#     for train_idx, test_idx in skf.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
        
#         estimator_clone = clone(estimator)
#         estimator_clone.fit(X_train, y_train)
#         y_pred = estimator_clone.predict(X_test)
#         score = hmetrics.h_f1_score(y_test, y_pred)
#         logging.info(f"Fold {i + 1} - Accuracy: {score}")
#         i += 1
        
#         scores.append(score)
    
#     return np.array(scores)


# def cross_val_predict(estimator, X, y, path, n_splits=5, random_state=42):
#     unique_classes, class_counts = np.unique(y, return_counts=True)
#     valid_classes = unique_classes[class_counts >= n_splits]

#     mask = np.isin(y, valid_classes)
#     X = X[mask]
#     y = y[mask]
#     y = LabelEncoder().fit_transform(y)
#     y_preds = np.zeros_like(y)
    
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     i = 0

#     for train_idx, test_idx in skf.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
        
#         estimator_clone = clone(estimator)
#         estimator_clone.fit(X_train, y_train)
#         y_pred = estimator_clone.predict(X_test)
#         score = accuracy_score(y_test, y_pred)
#         logging.info(f"Fold {i + 1} - Accuracy: {score}")

#         y_preds[test_idx] = y_pred
#         i += 1
    
#     utils.dump_parquet(data=y_preds, path=path)

#     accuracy = accuracy_score(y, y_preds)
#     logging.info(f"Accuracy: {accuracy}")
#     mcc = matthews_corrcoef(y, y_preds)
#     logging.info(f"MCC: {mcc}")
#     report = classification_report(y, y_preds)
#     logging.info(f"\n{report}")

#     return report, mcc


# def h_cross_val_predict(estimator, X, y, path, n_splits=5, random_state=42):
#     unique_classes, class_counts = np.unique(y, return_counts=True)
#     valid_classes = unique_classes[class_counts >= n_splits]

#     mask = np.isin(y, valid_classes)
#     X = X[mask]
#     y = y[mask]
#     y_preds = np.zeros_like(y)
#     y_normalized = list(map(get_hierarchy, y))
#     y_normalized = normalize_hierarchies(y_normalized)
#     y_normalized = np.array(y_normalized)
    
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     i = 0

#     for train_idx, test_idx in skf.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y_normalized[train_idx], y_normalized[test_idx]
        
#         estimator_clone = clone(estimator)
#         estimator_clone.fit(X_train, y_train)
#         y_pred = estimator_clone.predict(X_test)
#         score = hmetrics.h_f1_score(y_test, y_pred)
#         logging.info(f"Fold {i + 1} - Accuracy: {score}")

#         y_preds[test_idx] = y_pred
#         i += 1
    
#     utils.dump_parquet(data=y_preds, path=path)

#     accuracy = accuracy_score(y, y_preds)
#     logging.info(f"Accuracy: {accuracy}")
#     mcc = matthews_corrcoef(y, y_preds)
#     logging.info(f"MCC: {mcc}")
#     report = classification_report(y, y_preds)
#     logging.info(f"\n{report}")

#     return report, mcc
