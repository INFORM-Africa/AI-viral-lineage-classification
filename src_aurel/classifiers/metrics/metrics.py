"""
This script originates from the HiClass library (https://github.com/scikit-learn-contrib/hiclass/) 
Miranda, F.M., Köehnecke, N. and Renard, B.Y. (2023), HiClass: a Python Library for Local Hierarchical
Classification Compatible with Scikit-learn, Journal of Machine Learning Research, 24(29), pp. 1-17.
Available at: https://jmlr.org/papers/v24/21-1518.html.

Implements helper functions to compute hierarchical evaluation metrics.
"""

import numpy as np
from sklearn.utils import check_array

from ..abs_hierarchical_classifier import make_leveled


def _validate_input(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    y_pred = make_leveled(y_pred)
    y_true = make_leveled(y_true)
    y_true = check_array(y_true, dtype=None, ensure_2d=False, allow_nd=True)
    y_pred = check_array(y_pred, dtype=None, ensure_2d=False, allow_nd=True)
    return y_true, y_pred


def h_precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "micro"):
    r"""
    Compute hierarchical precision score.

    Parameters y_true (np.array of shape (n_samples, n_levels)) : Ground truth (correct) labels. y_pred (np.array of
    shape (n_samples, n_levels)) : Predicted labels, as returned by a classifier. average ({"micro", "macro"}, str,
    default="micro") : Determines the type of averaging performed during the computation:
            
            - `micro`: The precision is computed by summing over all individual instances, :math:`\displaystyle{hP =
            \frac{\sum_{i=1}^{n}| \alpha_i \cap \beta_i |}{\sum_{i=1}^{n}| \alpha_i |}}`, where :math:`\alpha_i` is
            the set consisting of the most specific classes predicted for test example :math:`i` and all their
            ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test
            example :math:`i` and all their ancestors, with summations computed over all test examples.

            - `macro`: The precision is computed for each instance and then averaged, :math:`\displaystyle{hP =
            \frac{\sum_{ i=1}^{n}hP_{i}}{n}}`, where :math:`\alpha_i` is the set consisting of the most specific
            classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the
            set containing the true most specific classes of test example :math:`i` and all their ancestors.

    Returns
        precision (float) : What proportion of positive identifications was actually correct?
    """
    y_true, y_pred = _validate_input(y_true, y_pred)
    functions = {
        "micro": _precision_micro,
        "macro": _precision_macro,
    }
    return functions[average](y_true, y_pred)


def _precision_micro(y_true: np.ndarray, y_pred: np.ndarray):
    precision_micro = {
        1: _precision_micro_1d,
        2: _precision_micro_2d,
        3: _precision_micro_3d,
    }
    return precision_micro[y_true.ndim](y_true, y_pred)


def _precision_micro_1d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = {ground_truth}
        ground_truth_set.discard("")
        predicted_set = {prediction}
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(predicted_set)
    return sum_intersection / sum_prediction_and_ancestors


def _precision_micro_2d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set(ground_truth)
        ground_truth_set.discard("")
        predicted_set = set(prediction)
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(predicted_set)
    return sum_intersection / sum_prediction_and_ancestors


def _precision_micro_3d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for row_ground_truth, row_prediction in zip(y_true, y_pred):
        ground_truth_set = set()
        predicted_set = set()
        for ground_truth, prediction in zip(row_ground_truth, row_prediction):
            ground_truth_set.update(ground_truth)
            predicted_set.update(prediction)
        ground_truth_set.discard("")
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(predicted_set)
    return sum_intersection / sum_prediction_and_ancestors


def _precision_macro(y_true: np.ndarray, y_pred: np.ndarray):
    return _compute_macro(y_true, y_pred, _precision_micro)


def h_recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "micro"):
    r"""
    Compute hierarchical recall score.

    Parameters y_true (np.array of shape (n_samples, n_levels)) : Ground truth (correct) labels. y_pred (np.array of
    shape (n_samples, n_levels)) : Predicted labels, as returned by a classifier. average ({"micro", "macro"}, str,
    default="micro") : Determines the type of averaging performed during the computation:

            - `micro`: The precision is computed by summing over all individual instances, :math:`\displaystyle{hP =
            \frac{\sum_{i=1}^{n}| \alpha_i \cap \beta_i |}{\sum_{i=1}^{n}| \alpha_i |}}`, where :math:`\alpha_i` is
            the set consisting of the most specific classes predicted for test example :math:`i` and all their
            ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test
            example :math:`i` and all their ancestors, with summations computed over all test examples.

            - `macro`: The precision is computed for each instance and then averaged, :math:`\displaystyle{hP =
            \frac{\sum_{i=1}^{n}hP_{i}}{n}}`, where :math:`\alpha_i` is the set consisting of the most specific
            classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the
            set containing the true most specific classes of test example :math:`i` and all their ancestors.

    Returns
    recall (float) : What proportion of actual positives was identified correctly?
    """
    y_true, y_pred = _validate_input(y_true, y_pred)
    functions = {
        "micro": _recall_micro,
        "macro": _recall_macro,
    }
    return functions[average](y_true, y_pred)


def _recall_micro(y_true: np.ndarray, y_pred: np.ndarray):
    recall_micro = {
        1: _recall_micro_1d,
        2: _recall_micro_2d,
        3: _recall_micro_3d,
    }
    return recall_micro[y_true.ndim](y_true, y_pred)


def _recall_micro_1d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = {ground_truth}
        ground_truth_set.discard("")
        predicted_set = {prediction}
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            ground_truth_set
        )
    recall = sum_intersection / sum_prediction_and_ancestors
    return recall


def _recall_micro_2d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set(ground_truth)
        ground_truth_set.discard("")
        predicted_set = set(prediction)
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            ground_truth_set
        )
    recall = sum_intersection / sum_prediction_and_ancestors
    return recall


def _recall_micro_3d(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for row_ground_truth, row_prediction in zip(y_true, y_pred):
        ground_truth_set = set()
        predicted_set = set()
        for ground_truth, prediction in zip(row_ground_truth, row_prediction):
            ground_truth_set.update(ground_truth)
            predicted_set.update(prediction)
        ground_truth_set.discard("")
        predicted_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(predicted_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            ground_truth_set
        )
    recall = sum_intersection / sum_prediction_and_ancestors
    return recall


def _recall_macro(y_true: np.ndarray, y_pred: np.ndarray):
    return _compute_macro(y_true, y_pred, _recall_micro)


def h_f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "micro"):
    r"""
    Compute hierarchical f-score.

    Parameters y_true (np.array of shape (n_samples, n_levels)) : Ground truth (correct) labels. y_pred (np.array of
    shape (n_samples, n_levels)) : Predicted labels, as returned by a classifier. average ({"micro", "macro"}, str,
    default="micro") : Determines the type of averaging performed during the computation:
            
            - `micro`: The precision is computed by summing over all individual instances, :math:`\displaystyle{hP =
            \frac{\sum_{i=1}^{n}| \alpha_i \cap \beta_i |}{\sum_{i=1}^{n}| \alpha_i |}}`, where :math:`\alpha_i` is
            the set consisting of the most specific classes predicted for test example :math:`i` and all their
            ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test
            example :math:`i` and all their ancestors, with summations computed over all test examples.

            - `macro`: The precision is computed for each instance and then averaged, :math:`\displaystyle{hP =
            \frac{\sum_{i=1}^{n}hP_{i}}{n}}`, where :math:`\alpha_i` is the set consisting of the most specific
            classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the
            set containing the true most specific classes of test example :math:`i` and all their ancestors.
    
    Returns
        f1 (float) : Weighted average of the precision and recall
    """
    y_true, y_pred = _validate_input(y_true, y_pred)
    functions = {
        "micro": _f_score_micro,
        "macro": _f_score_macro,
    }
    return functions[average](y_true, y_pred)


def _f_score_micro(y_true: np.ndarray, y_pred: np.ndarray):
    precision = h_precision_score(y_true, y_pred)
    rec = h_recall_score(y_true, y_pred)
    return 2 * precision * rec / (precision + rec)


def _f_score_macro(y_true: np.ndarray, y_pred: np.ndarray):
    return _compute_macro(y_true, y_pred, _f_score_micro)


def _compute_macro(y_true: np.ndarray, y_pred: np.ndarray, _micro_function):
    overall_sum = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        sample_score = _micro_function(np.array([ground_truth]), np.array([prediction]))
        overall_sum = overall_sum + sample_score
    return overall_sum / len(y_true)


def h_classification_report(y_true, y_pred):
    r"""
    Build a text report showing the main classification metrics.

    Parameters y_true (np.array of shape (n_samples, n_levels)) : Ground truth (correct) labels. y_pred (np.array of
    shape (n_samples, n_levels)) : Predicted labels, as returned by a classifier.

    Returns
        report (str) : Text summary of the precision, recall, and f1-score.
    """
    precision = h_precision_score(y_true, y_pred)
    recall = h_recall_score(y_true, y_pred)
    f1 = h_f1_score(y_true, y_pred)
    report = f"Precision: {precision}\nRecall: {recall}\nF1-score: {f1}\n"
    return report
