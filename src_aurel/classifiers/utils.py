import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn import metrics
import numpy as np

def is_terminal_node(node:str, hierarchy:dict):
    """
    Check if a node is a terminal node in the hierarchy
    Parameters
        node (str): Node to check
        hierarchy (dict): Hierarchy of the nodes
    Returns
        bool: True if the node is a terminal node, False otherwise
    """

    parent = get_parent(node)
    if parent in hierarchy:
        if node not in hierarchy[parent]:
            raise ValueError(f"Node {node} not found in hierarchy")
        
        if node not in hierarchy:
            return True
        
        return False
    raise ValueError(f"Node {node} not found in hierarchy")

def is_child_of(child:str, parent:str):
    """
    Check if a node is a child of another node in the hierarchy
    Parameters
        child (str): Child node
        parent (str): Parent node
    Returns
        bool: True if the child is a child of the parent, False otherwise
    """

    child_path = child.split(".")
    parent_path = parent.split(".")
    return all(x == y for x, y in zip(child_path, parent_path)) and child != parent

def is_parent_of(parent, child):
    """
    Check if a node is a parent of another node in the hierarchy
    Parameters
        parent (str): Parent node
        child (str): Child node
    Returns
        bool: True if the parent is a parent of the child, False otherwise
    """

    return is_child_of(child, parent)

def get_sibling_nodes(node, hierarchy):
    """
    Get the sibling nodes of a node in the hierarchy
    Parameters
        node (str): Node to get the siblings
        hierarchy (dict): Hierarchy of the nodes
    Returns
        set: Sibling nodes of the node
    """

    parent = get_parent(node)
    return hierarchy[parent] - {node}

def get_parent(node):
    """
    Get the parent node of a node
    Parameters
        node (str): Node to get the parent
    Returns
        str: Parent node of the node
    """

    parent = ".".join(node.split(".")[:-1])
    if parent == "":
        return "root"
    return parent

def save_learning_curves(model_name, history, plot_dir, perf_dir, plot_size=(10, 10)):
    """
    This function saves the learning curves of the deep learning models.
    Parameters
        history (keras.callbacks.History): Model history
        model_performance_dir (str): Directory to save the model performance metrics
        plot_dir (str): Directory to save models performances plots
        plot_size (tuple): Size of the plot
    Returns
        None
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plot_size)
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set(title='Model accuracy', ylabel='accuracy', xlabel='epoch')
    ax[0].legend(['Training', 'Validation'], loc='upper left')
    ax[0].grid()

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set(title='Model loss', ylabel='loss', xlabel='epoch')
    ax[1].legend(['Training', 'Validation'], loc='upper left')
    ax[1].grid()

    fig.savefig(f"{plot_dir}{model_name}_LC.pdf", format='pdf')
    np.save(f"{perf_dir}{model_name}history.npy", history.history)


def save_performance_plots(model_name: str, plot_dir: str, size: Tuple[int, int], conf_mat, fpr, tpr, roc_auc):
    """
    This function saves the performance plots of the models.
    Parameters
        model_name (str): Name of the model
        plot_dir (str): Directory to save the model performance plots
        size (tuple): Size of the plot
        conf_mat (np.ndarray): Confusion matrix
        fpr (np.ndarray): False positive rate
        tpr (np.ndarray): True positive rate
        roc_auc (float): Area under the curve
    Returns
        None
    """
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
        plt.savefig(f'{plot_dir}{model_name}_ROC.pdf', format='pdf')

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
            xticklabels=['No-bird Call', 'Bird Call'],
            yticklabels=['No-bird Call', 'Bird Call']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        # plt.title('CNN ClassiConfusion Matrix')
        plt.savefig(f'{plot_dir}{model_name}_CM.pdf', format='pdf')

    plot_roc_curve()
    plot_confusion_matrix()


def save_performance_metrics(model_name, plot_dir, perf_dir, size, Y_test, Y_preds, Y_probs, verbose=False):
    """
    This function saves the performance metrics of the models.
    Parameters
        model_name (str): Name of the model
        perf_dir (str): Directory to save the model performance metrics
        plot_dir (str): Directory to save the model performance plots
        size (tuple): Size of the plot
        Y_test (np.ndarray): Ground truth labels
        Y_preds (np.ndarray): Predicted labels
        Y_probs (np.ndarray): Predicted probabilities
        verbose (bool): Whether to print the performance metrics
    Returns
        None
    """
    accuracy = metrics.accuracy_score(Y_test, Y_preds)
    precision = metrics.precision_score(Y_test, Y_preds)
    recall = metrics.recall_score(Y_test, Y_preds)
    f1 = metrics.f1_score(Y_test, Y_preds)

    with open(f"{perf_dir}{model_name}.logs", 'w') as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1 Score: {f1:.2f}\n")
        file.close()

    print("Performance metrics saved successfully!")

    fpr, tpr, _ = metrics.roc_curve(Y_test, Y_probs)
    roc_auc = metrics.auc(fpr, tpr)
    conf_mat = metrics.confusion_matrix(Y_test, Y_preds)
    np.save(perf_dir + model_name + '_y_pred.npy', Y_preds)
    save_performance_plots(model_name, plot_dir, size, conf_mat, fpr, tpr, roc_auc)
    print("Performance plots saved successfully!")

    roc_data = {'fpr': fpr, 'tpr': tpr}

    try:
        with open(f"{perf_dir}{model_name}_ROC_data.txt", 'w') as file:
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
