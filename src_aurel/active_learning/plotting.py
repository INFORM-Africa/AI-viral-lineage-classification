import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def lags_histogram(lags, bins, xlabel=None, ylabel=None, title=None, save=True):
    """
    Plot histogram of lags
    """
    plt.hist(lags, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save:
        plt.savefig('lags_histogram.pdf', format='pdf', dpi=300)

    plt.show()

def lags_cdf(lags, xlabel=None, ylabel=None, title=None, save=True):
    """
    Plot CDF of lags
    """
    time_lags_sorted = np.sort(lags)
    cdf = np.arange(len(time_lags_sorted)) / float(len(time_lags_sorted))
    plt.plot(time_lags_sorted, cdf)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save:
        plt.savefig('lags_cdf.pdf', format='pdf', dpi=300)

    plt.show()

def lags_boxplot(lags, xlabel=None, title=None, save=True, figsize=(8, 6)):
    """
    Plot boxplot of lags
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=lags)
    plt.title(title)
    plt.xlabel(xlabel)

    if save:
        plt.savefig('lags_boxplot.pdf', format='pdf', dpi=300)

    plt.show()

def variants_detection_timeline(variants, traditional_detection, model_detection, figsize=(10, 5), save=True):
    plt.figure(figsize=figsize)
    plt.plot(variants, traditional_detection, label='Traditional Detection', marker='o')
    plt.plot(variants, model_detection, label='Model Detection', marker='o')
    plt.title('Variant Detection Timeline')
    plt.xlabel('Variant')
    plt.ylabel('Detection Date')
    plt.legend()

    if save:
        plt.savefig('variants_detection_timeline.pdf', format='pdf', dpi=300)

    plt.show()
