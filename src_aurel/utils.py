import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def load_settings(path: str = "settings.json"):
    """
    This function reads the settings from the settings.json file.
    Returns
        settings (dict): The settings dictionary
    """
    with open(path, 'r') as settings_file:
        settings = json.load(settings_file)
        
    return settings

def save_settings(settings: dict, path: str = "settings.json"):
    """
    This function saves the settings to the settings.json file.
    Parameters
        settings (dict): The settings dictionary
    Returns
        None
    """
    with open(path, 'w') as settings_file:
        json.dump(settings, settings_file, indent=4)

def load_feature_params(path: str = "feature_params.json"):
    with open(path, 'r') as params_file:
        params = json.load(params_file)
        
    return params

def dump_parquet(data, path):
    """
    This function saves the data to a parquet file.
    Parameters
        data (pd.DataFrame): The data to be saved
        path (str): The path to save the data
    Returns
        None
    """
    table = pa.table(pd.DataFrame(data))
    pq.write_table(table, path)

def read_parquet_to_np(path):
    """
    This function reads the data from a parquet file.
    Parameters
        path (str): The path to read the data
    Returns
        data (pd.DataFrame): The read data
    """
    table = pq.read_table(path)
    data = table.to_pandas().to_numpy()
    return data
