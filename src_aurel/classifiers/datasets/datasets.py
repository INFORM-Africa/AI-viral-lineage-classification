"""
This script originates from the HiClass library (https://github.com/scikit-learn-contrib/hiclass/) 
Miranda, F.M., Köehnecke, N. and Renard, B.Y. (2023), HiClass: a Python Library for Local Hierarchical
Classification Compatible with Scikit-learn, Journal of Machine Learning Research, 24(29), pp. 1-17.
Available at: https://jmlr.org/papers/v24/21-1518.html.

Datasets util for downloading and maintaining sample datasets.
"""

import requests
import pandas as pd
import os
import tempfile
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use temp directory to store cached datasets
CACHE_DIR = tempfile.gettempdir()

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Dataset urls
PLATYPUS_URL = ("https://gist.githubusercontent.com/ashishpatel16/9306f8ed3ed101e7ddcb519776bcbd80/raw"
                "/1152c0b9613c2bda144a38fc4f74b5fe12255f4d/platypus_diseases.csv")


def _download_file(url, destination):
    """Download file from given URL to specified destination."""
    try:
        response = requests.get(url)
        # Raise HTTPError if response code is not OK
        response.raise_for_status()
        with open(destination, "wb") as f:
            f.write(response.content)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download file from {url}: {str(e)}")


def load_platypus(test_size=0.3, random_state=42):
    """
    Load platypus diseases dataset.

    Parameters test_size (float, default=0.3) : The proportion of the dataset to include in the test split.
    random_state (int or None, default=42) : Controls the randomness of the dataset. Pass an int for reproducible
        output across multiple function calls.

    Returns 
        list : List containing train-test split of inputs.

    Raises
        RuntimeError : If failed to access or process the dataset.

    Examples
        >>> from classifiers.datasets import load_platypus
        >>> X_train, X_test, Y_train, Y_test = load_platypus()
        >>> X_train[:3]
            fever  diarrhea  stomach pain  skin rash  cough  sniffles  short breath  headache  size
        220   37.8         0             3          5      1         1             0         2  27.6
        539   37.2         0             6          1      1         1             0         3  28.4
        326   39.9         0             2          5      1         1             1         2  30.7
        >>> X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
        (572, 9) (246, 9) (572,) (246,)
    """
    dataset_name = "platypus_diseases.csv"
    cached_file_path = os.path.join(CACHE_DIR, dataset_name)

    logger.info("Downloading platypus diseases dataset..")
    print("Downloading platypus diseases dataset..")

    # Check if the file exists in the cache
    if not os.path.exists(cached_file_path):
        try:
            logger.info("Downloading platypus diseases dataset..")
            _download_file(PLATYPUS_URL, cached_file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to access or download dataset: {str(e)}")

    data = pd.read_csv(cached_file_path).fillna(" ")
    X = data.drop(["label"], axis=1)
    y = pd.Series([eval(val) for val in data["label"]])

    # Return tuple (X_train, X_test, y_train, y_test)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
