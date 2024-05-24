import utils, os, logging
from classifiers import metrics as hmetrics
from model_selection import train_test_split
from preprocessing.read_data import normalize_hierarchies
from classifiers import LocalClassifierPerLevel, LocalClassifierPerNode, LocalClassifierPerParentNode
from datetime import datetime

# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def get_timestamp():
    now = datetime.now()
    return int(now.timestamp())

def train_hmodel(arch:str, clf, features, labels, features_name, clf_name, reports_dir, test_size=0.2):
    timestamp = get_timestamp()
    reports_filename = os.path.join(reports_dir, f'h_{arch}_{clf_name}_{features_name}_{timestamp}.txt')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    if arch == 'lcpl':
        logging.info(f"Training Local Classifier Per Level with {features_name} features")
        h_clf = LocalClassifierPerLevel(
            local_classifier=clf, 
            replace_classifiers=False,
            n_jobs=-1,
        )
    elif arch == 'lcpn':
        logging.info(f"Training Local Classifier Per Node with {features_name} features")
        h_clf = LocalClassifierPerNode(
            local_classifier=clf,
            replace_classifiers=False,
            n_jobs=-1,
            binary_policy="inclusive",
        )
    elif arch == 'lcppn':
        logging.info(f"Training Local Classifier Per Parent Node with {features_name} features")
        h_clf = LocalClassifierPerParentNode(
            local_classifier=clf,
            n_jobs=-1,
            replace_classifiers=False,
        )
    else:
        raise ValueError(f"Invalid hierarchical architecture: {arch}")

    # Train local classifier per level
    h_clf.fit(X_train, y_train)

    # Predict
    predictions = h_clf.predict(X_test)

    # Evaluate
    summary = hmetrics.h_classification_report(y_test, predictions)

    # Save logs
    with open(reports_filename, 'a') as file:
        file.write(summary)
    
    logging.info(f"Training completed with {summary}")
    logging.info(f"Logs saved to {reports_filename}")


def train_hmodel_with_features(htype:str, model:str, X, y, feature:str, reports_dir:str):
    if model == 'rf':
        logging.info(f"Training Random Forest with {feature} features")
        clf = RandomForestClassifier()
    elif model == 'xgb':
        logging.info(f"Training XGBoost with {feature} features")
        clf = XGBClassifier()
    elif model == 'lgbm':
        logging.info(f"Training LightGBM with {feature} features")
        clf = LGBMClassifier()
    elif model == 'cb':
        logging.info(f"Training CatBoost with {feature} features")
        clf = CatBoostClassifier()
    else:
        raise ValueError(f"Invalid model: {model}")
    
    train_hmodel(htype, clf, X, y, feature, model, reports_dir)
    return


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    settings = utils.load_settings(path="src_aurel/settings.json")
    params = utils.load_feature_params(path="src_aurel/feature_params.json")

    logging.info("Loading settings")
    features_dir = settings["features_dir"]
    cleaned_data_dir = settings["cleaned_data_dir"]
    reports_dir = settings["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    n_trials = settings["optuna_trials"]
    hierachical_architectures = settings["hierachical_architectures"]
    train_models = settings["train_models"]

    if "training_completed" in settings:
        training_completed = settings["training_completed"]
    else:
        training_completed = []

    logging.info(f"Loading labels file flat_labels.parquet")
    y = utils.read_parquet_to_np(os.path.join(cleaned_data_dir, "hierarchical_labels.parquet"))
    y = normalize_hierarchies(y[:, 0])

    for hmodel in hierachical_architectures:
        for model in train_models:
            if model not in ['rf', 'xgb', 'lgbm', 'cb']:
                raise ValueError(f"Invalid model: {model}")
            for key, config in params.items():
                if key == 'kmer':
                    ks = config['k']
                    for k in ks:
                        train_key = f"{hmodel}::{model}::kmer_{k}"
                        if train_key not in training_completed:
                            logging.info(f"Loading features file kmer_{k}_features.parquet")
                            features_path = os.path.join(features_dir, f"kmer_{k}_features.parquet")
                            X = utils.read_parquet_to_np(features_path)
                            train_hmodel_with_features(
                                htype=hmodel,
                                model=model,
                                X=X,
                                y=y,
                                feature=f"kmer_{k}",
                                reports_dir=reports_dir,
                            )
                            training_completed.append(train_key)
                            settings["training_completed"] = training_completed
                            utils.save_settings(settings, path="src_aurel/settings.json")
                            logging.info(f"Trained hmodels updated with {train_key}")
                        else:
                            logging.info(f"Skipping {train_key} as it is already completed")
                        
                elif key == 'fcgr':
                    ress = config['resolution']
                    for res in ress:
                        train_key = f"{hmodel}::{model}::fcgr_{res}"
                        if train_key not in training_completed:
                            logging.info(f"Loading features file fcgr_{res}_features.parquet")
                            features_path = os.path.join(features_dir, f"fcgr_{res}_features.parquet")
                            X = utils.read_parquet_to_np(features_path)
                            train_hmodel_with_features(
                                htype=hmodel,
                                model=model,
                                X=X,
                                y=y,
                                feature=f"fcgr_{res}",
                                reports_dir=reports_dir,
                            )
                            training_completed.append(train_key)
                            settings["training_completed"] = training_completed
                            utils.save_settings(settings, path="src_aurel/settings.json")
                            logging.info(f"Trained hmodels updated with {train_key}")
                        else:
                            logging.info(f"Skipping {train_key} as it is already completed")
                        
                elif key == 'murugaiah':
                    train_key = f"{hmodel}::{model}::murugaiah"
                    if train_key not in training_completed:
                        logging.info(f"Loading features file murugaiah_features.parquet")
                        features_path = os.path.join(features_dir, f"murugaiah_features.parquet")
                        X = utils.read_parquet_to_np(features_path)
                        train_hmodel_with_features(
                            htype=hmodel,
                            model=model,
                            X=X,
                            y=y,
                            feature=f"murugaiah",
                            reports_dir=reports_dir,
                        )
                        training_completed.append(train_key)
                        settings["training_completed"] = training_completed
                        utils.save_settings(settings, path="src_aurel/settings.json")
                        logging.info(f"Trained hmodels updated with {train_key}")
                    else:
                        logging.info(f"Skipping {train_key} as it is already completed")
                    
                else:
                    raise ValueError(f"Invalid key: {key}")
