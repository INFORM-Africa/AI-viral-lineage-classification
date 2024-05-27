import logging
import pandas as pd
import numpy as np
import utils, os
from active_learning import ActiveLearning, plotting
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def run_detection_simulation(detector:ActiveLearning, X:np.ndarray, y:np.ndarray, dates:np.ndarray, start_date:str, step:int, features:str, reports_dir:str):
    detector.fit_all_query_strategies(
        X=X, 
        y=y,
        dates=dates,
        initial_date=start_date,
        timestep=step,
        output_dir=reports_dir,
        features=features,
    )
    return


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    settings = utils.load_settings(path="src_aurel/settings.json")
    params = utils.load_feature_params(path="src_aurel/feature_params.json")

    logging.info("Loading settings")
    features_dir = settings["features_dir"]
    cleaned_data_dir = settings["cleaned_data_dir"]
    reports_dir = settings["reports_dir"]
    detection_start_date = settings["detection_start_date"]
    detection_step = settings["detection_step"]
    os.makedirs(reports_dir, exist_ok=True)

    if "variants_detection_completed" in settings:
        variants_detection_completed = settings["variants_detection_completed"]
    else:
        variants_detection_completed = []

    logging.info(f"Loading labels file flat_labels.parquet")
    le = LabelEncoder()
    y = utils.read_parquet_to_np(os.path.join(cleaned_data_dir, "flat_labels.parquet"))
    y = le.fit_transform(y.flatten())

    logging.info("Loading dates")
    df = pd.read_parquet(os.path.join(cleaned_data_dir, "cleaned_dataset.parquet"))
    dates = df['date'].values
    del df

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
                detection_key = f"kmer_{k}"
                if detection_key not in variants_detection_completed:
                    logging.info(f"Loading features file kmer_{k}_features.parquet")
                    features_path = os.path.join(features_dir, f"kmer_{k}_features.parquet")
                    X = utils.read_parquet_to_np(features_path)
                    run_detection_simulation(
                        detector=variant_detector,
                        X=X, 
                        y=y, 
                        dates=dates,
                        step=detection_step,
                        start_date=detection_start_date,
                        features=f"kmer_{k}", 
                        reports_dir=reports_dir
                    )
                    variants_detection_completed.append(detection_key)
                    settings["variants_detection_completed"] = variants_detection_completed
                    utils.save_settings(settings, path="src_aurel/settings.json")
                    logging.info(f"Detection simulation completed for {detection_key}")
                else:
                    logging.info(f"Skipping {detection_key} as it is already completed")
                
        elif key == 'fcgr':
            ress = config['resolution']
            for res in ress:
                detection_key = f"fcgr_{res}"
                if detection_key not in variants_detection_completed:
                    logging.info(f"Loading features file fcgr_{res}_features.parquet")
                    features_path = os.path.join(features_dir, f"fcgr_{res}_features.parquet")
                    X = utils.read_parquet_to_np(features_path)
                    run_detection_simulation(
                        detector=variant_detector,
                        X=X, 
                        y=y, 
                        dates=dates,
                        step=detection_step,
                        start_date=detection_start_date,
                        features=f"fcgr_{res}", 
                        reports_dir=reports_dir
                    )
                    variants_detection_completed.append(detection_key)
                    settings["variants_detection_completed"] = variants_detection_completed
                    utils.save_settings(settings, path="src_aurel/settings.json")
                    logging.info(f"Detection simulation completed for {detection_key}")
                else:
                    logging.info(f"Skipping {detection_key} as it is already completed")
                
        elif key == 'murugaiah':
            detection_key = f"murugaiah"
            if detection_key not in variants_detection_completed:
                logging.info(f"Loading features file murugaiah_features.parquet")
                features_path = os.path.join(features_dir, f"murugaiah_features.parquet")
                X = utils.read_parquet_to_np(features_path)
                run_detection_simulation(
                    detector=variant_detector,
                    X=X, 
                    y=y, 
                    dates=dates,
                    step=detection_step,
                    start_date=detection_start_date,
                    features=f"murugaiah", 
                    reports_dir=reports_dir
                )
                variants_detection_completed.append(detection_key)
                settings["variants_detection_completed"] = variants_detection_completed
                utils.save_settings(settings, path="src_aurel/settings.json")
                logging.info(f"Detection simulation completed for {detection_key}")
            else:
                logging.info(f"Skipping {detection_key} as it is already completed")
            
        else:
            raise ValueError(f"Invalid key: {key}")


