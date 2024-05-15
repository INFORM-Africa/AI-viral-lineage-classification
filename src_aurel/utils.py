import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn import metrics
import numpy as np
import json, os

def load_settings():
    """
    This function reads the settings from the settings.json file.
    Returns
        settings (dict): The settings dictionary
    """
    with open("src_aurel/settings.json", 'r') as settings_file:
        settings = json.load(settings_file)

    if 'created' not in settings:
        for _, path in settings.items():
            if not os.path.exists(path):
                os.makedirs(path)

        settings['created'] = True

        with open("settings.json", 'w') as settings_file:
            json.dump(settings, settings_file, indent=4)
        
    return settings


def save_performance_plots(model_name: str, size: Tuple[int, int], conf_mat, fpr, tpr, roc_auc):
    """
    This function saves the performance plots of the models.
    Parameters
        model_name (str): Name of the model
        size (tuple): Size of the plot
        conf_mat (np.ndarray): Confusion matrix
        fpr (np.ndarray): False positive rate
        tpr (np.ndarray): True positive rate
        roc_auc (float): Area under the curve
    Returns
        None
    """
    with open("settings.json", 'r') as settings_file:
        settings = json.load(settings_file)

    plots_dir = settings["plot_dir"]

    def plot_roc_curve():
        """
        Plot the ROC curve.
        """
        plt.figure(figsize=size)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve for {model_name} Classifier')
        plt.legend(loc='lower right')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(f'{plots_dir}{model_name}_ROC.pdf', format='pdf')

    def plot_confusion_matrix():
        """
        Plot the confusion matrix.
        """
        plt.figure(figsize=size)
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
        plt.savefig(f'{plots_dir}{model_name}_CM.pdf', format='pdf')

    plot_roc_curve()
    plot_confusion_matrix()


def save_performance_metrics(model_name, size, Y_test, Y_preds, Y_probs, verbose=False):
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

    with open("settings.json", 'r') as settings_file:
        settings = json.load(settings_file)

    plots_dir = settings["plot_dir"]
    reports_dir = settings["reports_dir"]

    accuracy = metrics.accuracy_score(Y_test, Y_preds)
    precision = metrics.precision_score(Y_test, Y_preds)
    recall = metrics.recall_score(Y_test, Y_preds)
    f1 = metrics.f1_score(Y_test, Y_preds)

    with open(f"{reports_dir}{model_name}.logs", 'w') as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1 Score: {f1:.2f}\n")
        file.close()

    print("Performance metrics saved successfully!")

    fpr, tpr, _ = metrics.roc_curve(Y_test, Y_probs)
    roc_auc = metrics.auc(fpr, tpr)
    conf_mat = metrics.confusion_matrix(Y_test, Y_preds)
    np.save(reports_dir + model_name + '_y_pred.npy', Y_preds)
    save_performance_plots(model_name, plots_dir, size, conf_mat, fpr, tpr, roc_auc)
    print("Performance plots saved successfully!")

    roc_data = {'fpr': fpr, 'tpr': tpr}

    try:
        with open(f"{reports_dir}{model_name}_ROC_data.txt", 'w') as file:
            file.write(str(roc_data))
    except Exception as e:
        print(f"Error: {e}")

    

    print("ROC data saved successfully!")

    if verbose:
        print(f"Accuracy: {accuracy:.2f}")
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(f'AUC: {roc_auc:.2f}')
