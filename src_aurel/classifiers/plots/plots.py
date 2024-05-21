import utils, os
import numpy as np
from sklearn import metrics
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def _plot_roc_curve(model: str, features:str, plots_dir:str, figsize: Tuple[int, int], fpr, tpr, roc_auc):
        """
        Plot the ROC curve.
        """
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve for {model_name} Classifier')
        plt.legend(loc='lower right')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        output_path = os.path.join(plots_dir, f'{model}-{features}-ROC.pdf')
        plt.savefig(output_path, format='pdf')
        plt.close()

def _plot_confusion_matrix(model: str, features:str,  plots_dir:str, figsize: Tuple[int, int], conf_mat):
    """
    Plot the confusion matrix.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        # xticklabels=['No-bird Call', 'Bird Call'],
        # yticklabels=['No-bird Call', 'Bird Call']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    output_path = os.path.join(plots_dir, f'{model}-{features}-CM.pdf')
    plt.savefig(output_path, format='pdf')
    plt.close()


def save_performance_metrics(model, features, Y_test, Y_preds):
    """
    This function saves the performance metrics of the models.
    Parameters
        model_name (str): Name of the model
        size (tuple): Size of the plot
        Y_test (np.ndarray): Ground truth labels
        Y_preds (np.ndarray): Predicted labels
        Y_probs (np.ndarray): Predicted probabilities
        verbose (bool): Whether to print the performance metrics
    Returns
        None
    """

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    settings = utils.load_settings(path="src_aurel/settings.json")
    reports_dir = settings["reports_dir"]

    assert os.path.exists(reports_dir), f"Path {reports_dir} does not exist!"

    accuracy = metrics.accuracy_score(Y_test, Y_preds)
    precision = metrics.precision_score(Y_test, Y_preds)
    recall = metrics.recall_score(Y_test, Y_preds)
    f1 = metrics.f1_score(Y_test, Y_preds)

    summary = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    logs_file_path = os.path.join(reports_dir, f"{model}-{features}.logs")

    with open(logs_file_path, 'w') as file:
        file.write(str(summary))

    logging.info(f"Performance metrics {model}::{features} saved to {logs_file_path}")


def plot_performance_curve(model, features, figsize, Y_test, Y_preds, Y_probs):
    settings = utils.load_settings(path="src_aurel/settings.json")
    plots_dir = settings["plot_dir"]
    reports_dir = settings["reports_dir"]

    assert os.path.exists(reports_dir), f"Path {reports_dir} does not exist!"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    fpr, tpr, _ = metrics.roc_curve(Y_test, Y_probs)
    roc_auc = metrics.auc(fpr, tpr)
    conf_mat = metrics.confusion_matrix(Y_test, Y_preds)
    # np.save(reports_dir + model + '_y_pred.npy', Y_preds)
    
    _plot_roc_curve(model, features, plots_dir, figsize, fpr, tpr, roc_auc)
    _plot_confusion_matrix(model, features, plots_dir, figsize, conf_mat)

    print("Performance plots saved successfully!")

    roc_summary = {'fpr': fpr, 'tpr': tpr}

    try:
        with open(f"{reports_dir}{model}_ROC_data.txt", 'w') as file:
            file.write(str(roc_summary))
    except Exception as e:
        logging.error(f"Error: {e}")
    
    print("ROC data saved successfully!")
