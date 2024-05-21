import logging
import requests, zipfile, io, os
import pandas as pd
from feature_extraction import ConventionalFeatures, MurugaiahFeatures
import utils
from preprocessing.read_data import extract_metadata, extract_sequences, remove_consensus_call_sequences
from preprocessing.read_data import clean_sequences, clean_collection_dates, get_hierarchy, merge_dfs


def download_and_save_data(url: str, output_dir:str) -> None:
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(output_dir)
    return

def should_create(output_dir:str) -> bool:
    return not os.path.exists(output_dir)

def clean_and_dump_data(input_dir, dump_dir) -> None:
    metadata = extract_metadata(input_dir)
    metadata = clean_collection_dates(metadata)

    sequences = extract_sequences(input_dir)
    sequences = clean_sequences(sequences)
    
    data = merge_dfs(metadata, sequences)
    data = remove_consensus_call_sequences(data)
    data["lineage_hierarchy"] = data["lineage"].apply(get_hierarchy)

    data.to_parquet(os.path.join(dump_dir, 'cleaned_dataset.parquet'), index=False)
    utils.dump_parquet(data['lineage'], os.path.join(dump_dir, 'flat_labels.parquet'))
    utils.dump_parquet(data['lineage_hierarchy'], os.path.join(dump_dir, 'hierarchical_labels.parquet'))
    return

def extract_and_dump_features(kind:str, sequences, output_dir, *args) -> None:
    if kind == 'kmer':
        filename = f'kmer_{args[0]}_features.parquet'
        if should_create(os.path.join(output_dir, filename)):
            features = ConventionalFeatures().extract_kmers_features(sequences, *args)
            utils.dump_parquet(features, os.path.join(output_dir, filename))
            logging.info(f"Features saved to {str(output_dir)}{filename}")
        else:
            logging.info(f"File {filename} already exists, skipping...")

    elif kind == 'fcgr':
        filename = f'fcgr_{args[0]}_features.parquet'
        if should_create(os.path.join(output_dir, filename)):
            features = ConventionalFeatures().extract_fcgr_features(sequences, *args)
            features = features.reshape((len(sequences), -1))
            utils.dump_parquet(features, os.path.join(output_dir, filename))
            logging.info(f"Features saved to {str(output_dir)}{filename}")
        else:
            logging.info(f"File {filename} already exists, skipping...")
        
    elif kind == 'murugaiah':
        filename = 'murugaiah_features.parquet'
        if should_create(os.path.join(output_dir, filename)):
            features = MurugaiahFeatures().extract_features(sequences, *args)
            utils.dump_parquet(features, os.path.join(output_dir, filename))
            logging.info(f"Features saved to {str(output_dir)}{filename}")
        else:
            logging.info(f"File {filename} already exists, skipping...")
    
    else:
        raise ValueError(f"Invalid kind: {kind}")

    return

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    url = "https://download1591.mediafire.com/vnfbz44a9zjgrMzzQvp_yyugINOzdoSwuLnMead0XC8dpkl-oLwHr45OjhI25NCALEoMGvXSo0OnvconWhLbczAwUHYcG81aULMxQQwwF6Je4GSLA-pnd0yZuatlMvk6qoY2MtvPI2ROSST2_09zv5GxAzQOCGwJ4Qe4pd03fA/cugn87w78xejegc/raw_data.zip"
    
    logging.info("Loading settings")
    settings = utils.load_settings(path="src_aurel/settings.json")
    raw_data_dir = settings['raw_data_dir']
    features_dir = settings['features_dir']
    dataset_dir = settings['dataset_dir']
    cleaned_data_dir = settings["cleaned_data_dir"]

    if should_create(raw_data_dir):
        logging.info("Downloading the raw data")
        download_and_save_data(url, output_dir=dataset_dir)
        logging.info("Download OK")
    else:
        logging.info("Raw data already exists, skipping...")

    if should_create(cleaned_data_dir):
        logging.info("Cleaning the raw data")
        os.makedirs(cleaned_data_dir, exist_ok=True)
        clean_and_dump_data(input_dir=raw_data_dir, dump_dir=cleaned_data_dir)
        logging.info("Cleaning OK")
    else:
        logging.info("Cleaned data already exists, skipping...")

    params = utils.load_feature_params(path="src_aurel/feature_params.json")
    os.makedirs(features_dir, exist_ok=True)
    logging.info(f"Reading cleaned data from {cleaned_data_dir}")
    df = pd.read_parquet(os.path.join(cleaned_data_dir, "cleaned_dataset.parquet"))

    sequences = df['sequence'].tolist()
    logging.info(f"Extracting features for {len(sequences)} sequences")

    for key, config in params.items():
        if key == 'kmer':
            ks = config['k']
            normalize = config['normalize']
            for k in ks:
                logging.info(f"Extracting {key} features with k={k}, normalize={normalize}")
                extract_and_dump_features('kmer', sequences, features_dir, k, normalize)
        elif key == 'fcgr':
            ress = config['resolution']
            for res in ress:
                logging.info(f"Extracting {key} features with resolution={res}")
                extract_and_dump_features('fcgr', sequences, features_dir, res)
        elif key == 'murugaiah':
            logging.info(f"Extracting {key} features")
            extract_and_dump_features('murugaiah', sequences, features_dir)
        else:
            raise ValueError(f"Invalid key: {key}")
    
    logging.info("Feature extraction OK")