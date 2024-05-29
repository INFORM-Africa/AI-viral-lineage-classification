import logging
import pandas as pd
import numpy as np
import utils, os
from active_learning import ActiveLearning
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def run_detection_simulation(detector:ActiveLearning, X:np.ndarray, y:np.ndarray, dates:np.ndarray, features:str):
    query_strategies = ['least_confident', 'margin_sampling', 'entropy']
    settings = utils.load_settings(path="src_aurel/settings.json")
    reports_dir = settings["reports_dir"]
    start_date = settings["detection_start_date"]
    step = settings["detection_step"]

    if "variants_detection_completed" in settings:
        variants_detection_completed = settings["variants_detection_completed"]
    else:
        variants_detection_completed = []

    for query_strategy in query_strategies:
        detection_key = f"{features}::{query_strategy}"

        if detection_key not in variants_detection_completed:
            logging.info(f"Running detection simulation for {features} with query strategy {query_strategy}")
            detection_dates, reported_lineages, missed_lineages = detector.fit(X, y, dates, start_date, step, query_strategy)
            
            reported_lineages_df = pd.DataFrame(reported_lineages)
            missed_lineages_df = pd.DataFrame(missed_lineages)
            detection_dates_df = pd.DataFrame(detection_dates)
            utils.write_detection_outputs(
                features=features,
                query_strategy=query_strategy,
                reported_lineages_df=reported_lineages_df,
                missed_lineages_df=missed_lineages_df,
                detection_dates_df=detection_dates_df,
                reports_dir=reports_dir
            )
            variants_detection_completed.append(detection_key)
            settings["variants_detection_completed"] = variants_detection_completed
            utils.save_settings(settings, path="src_aurel/settings.json")
            logging.info(f"Detection simulation completed for {detection_key}")
        else:
            logging.info(f"Skipping {detection_key} as it is already completed")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    settings = utils.load_settings(path="src_aurel/settings.json")
    params = utils.load_feature_params(path="src_aurel/feature_params.json")

    logging.info("Loading settings")
    features_dir = settings["features_dir"]
    cleaned_data_dir = settings["cleaned_data_dir"]

    logging.info(f"Loading labels file flat_labels.parquet")
    le = LabelEncoder()
    y = utils.read_parquet_to_np(os.path.join(cleaned_data_dir, "flat_labels.parquet"))
    y = le.fit_transform(y.flatten())

    logging.info("Loading dates")
    df = pd.read_parquet(os.path.join(cleaned_data_dir, "cleaned_dataset.parquet"))
    dates = df['date'].values

    base_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    variant_detector = ActiveLearning(
        base_estimator=base_estimator, 
        budget=10,
        decoder=le,
        lineages_df=df,
    )

    for key, config in params.items():
        if key == 'kmer':
            ks = config['k']
            for k in ks:
                logging.info(f"Loading features file kmer_{k}_features.parquet")
                features_path = os.path.join(features_dir, f"kmer_{k}_features.parquet")
                X = utils.read_parquet_to_np(features_path)
                run_detection_simulation(
                    detector=variant_detector,
                    X=X, 
                    y=y, 
                    dates=dates,
                    features=f"kmer_{k}",
                )
                logging.info(f"Detection simulation completed for kmer_{k}")
                
        elif key == 'fcgr':
            ress = config['resolution']
            for res in ress:
                logging.info(f"Loading features file fcgr_{res}_features.parquet")
                features_path = os.path.join(features_dir, f"fcgr_{res}_features.parquet")
                X = utils.read_parquet_to_np(features_path)
                run_detection_simulation(
                    detector=variant_detector,
                    X=X, 
                    y=y, 
                    dates=dates,
                    features=f"fcgr_{res}",
                )
                logging.info(f"Detection simulation completed for fcgr_{res}")
                
        elif key == 'murugaiah':
            logging.info(f"Loading features file murugaiah_features.parquet")
            features_path = os.path.join(features_dir, f"murugaiah_features.parquet")
            X = utils.read_parquet_to_np(features_path)
            run_detection_simulation(
                detector=variant_detector,
                X=X,
                y=y,
                dates=dates,
                features=f"murugaiah",
            )
            logging.info(f"Detection simulation completed for murugaiah")
            
        else:
            raise ValueError(f"Invalid key: {key}")


