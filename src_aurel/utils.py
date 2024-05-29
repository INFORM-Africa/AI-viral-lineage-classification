import json, os, ast
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def load_aliases(path: str = "alias_key.json"):
    """
    This function reads the aliases from the alias_key.json file.
    Arguments:
        path (str): The path to the alias_key.json file
    Returns
        aliases (dict): The aliases dictionary
    """
    with open(path, 'r') as alias_file:
        aliases = json.load(alias_file)
        
    return aliases


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

def read_best_hyperparameters(h_model:str, model:str, feature:str, reports_dir:str):
    file_path = os.path.join(reports_dir, f'{h_model}_{model}_{feature}.txt')

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Best hyperparameters:"):
                hyperparams_str = line.split("Best hyperparameters: ")[1].strip()
                hyperparams_dict = ast.literal_eval(hyperparams_str)
                return hyperparams_dict
        raise ValueError(f"Best hyperparameters not found in {file_path}")
    
def write_training_duration(h_model:str, model:str, feature:str, reports_dir:str, duration:float):
    file_path = os.path.join(reports_dir, f'training_durations.txt')
    with open(file_path, 'a') as file:
        file.write(f"{h_model}::{model}::{feature} ==> {duration:.4f} seconds\n")
    
    return

def write_detection_outputs(features:str, query_strategy:str, reports_dir:str, reported_lineages_df:pd.DataFrame, missed_lineages_df:pd.DataFrame, detection_dates_df:pd.DataFrame):
    reported_lineages_filename = os.path.join(reports_dir, f'{features}_reported_lineages_{query_strategy}.tsv')
    missed_lineages_filename = os.path.join(reports_dir, f'{features}_missed_lineages_{query_strategy}.tsv')
    detection_dates_filename = os.path.join(reports_dir, f'{features}_detection_dates_{query_strategy}.tsv')
    reported_lineages_df.to_csv(reported_lineages_filename, sep='\t', index=False)
    missed_lineages_df.to_csv(missed_lineages_filename, sep='\t', index=False)
    detection_dates_df.to_csv(detection_dates_filename, sep='\t', index=False)