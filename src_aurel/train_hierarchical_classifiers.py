import sys
import utils, os, logging, time
from model_selection import h_train_test_split_predict
from classifiers import LocalClassifierPerLevel, LocalClassifierPerNode, LocalClassifierPerParentNode

# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def train_model(clf, X, y, test_size, h_type, feature, clf_name, reports_dir):
    start_time = time.time()
    reports_filename = os.path.join(reports_dir, f'training_{h_type}_{clf_name}_{feature}.txt')

    if h_type == 'lcpl':
        logging.info(f"Training Local Classifier Per Level with {feature} features")
        h_clf = LocalClassifierPerLevel(
            local_classifier=clf, 
            replace_classifiers=False,
            n_jobs=-1,
        )
    elif h_type == 'lcpn':
        logging.info(f"Training Local Classifier Per Node with {feature} features")
        h_clf = LocalClassifierPerNode(
            local_classifier=clf,
            replace_classifiers=False,
            n_jobs=-1,
            binary_policy="inclusive",
        )
    elif h_type == 'lcppn':
        logging.info(f"Training Local Classifier Per Parent Node with {feature} features")
        h_clf = LocalClassifierPerParentNode(
            local_classifier=clf,
            n_jobs=-1,
            replace_classifiers=False,
        )
    else:
        raise ValueError(f"Invalid hierarchical architecture: {h_type}")

    report = h_train_test_split_predict(
        estimator=h_clf,
        X=X, 
        y=y, 
        test_size=test_size, 
        random_state=42
    )

    elapsed_time = time.time() - start_time

    # Save logs
    with open(reports_filename, 'a') as file:
        file.write(report)

    utils.write_training_duration(
        h_model=h_type,
        model=clf_name, 
        feature=feature,
        reports_dir=reports_dir,
        duration=elapsed_time,
    )
    
    logging.info(f"Training completed in {elapsed_time:.2f} with \n{report}")
    logging.info(f"Logs saved to {reports_filename}")


def train_model_with_features(htype:str, model:str, X, y, test_size:float, split:str, feature:str, reports_dir:str):
    file_path = os.path.join(reports_dir, f'{htype}_{model}_{split}_{feature}.txt')
    
    if not os.path.exists(file_path):
        logging.info(f"Best hyperparameters file not found: {file_path}")
        return
    
    model_params = utils.read_best_hyperparameters(
        h_model=htype,
        model=model,
        feature=feature,
        reports_dir=reports_dir,
        split=split,
    )

    if model == 'rf':
        logging.info(f"Training Random Forest with {feature} features")
        clf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **model_params
        )
    elif model == 'xgb':
        logging.info(f"Training XGBoost with {feature} features")
        clf = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            **model_params
        )
    elif model == 'lgbm':
        logging.info(f"Training LightGBM with {feature} features")
        clf = LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            **model_params
        )
    elif model == 'cb':
        logging.info(f"Training CatBoost with {feature} features")
        clf = CatBoostClassifier(
            random_state=42,
            n_jobs=-1,
            **model_params
        )
    else:
        raise ValueError(f"Invalid model: {model}")
    
    train_model(clf, X, y, test_size, htype, feature, model, reports_dir)
    return


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    settings = utils.load_settings(path="src_aurel/settings.json")

    logging.info("Loading settings")
    features_dir = settings["features_dir"]
    cleaned_data_dir = settings["cleaned_data_dir"]
    reports_dir = settings["reports_dir"]
    finetuned_models = settings["fine_tune_completed"]
    test_size = settings["test_size"]
    cv = settings["use_cross_validation"]
    split = "cv" if cv else "tts"

    if "training_completed" in settings:
        training_completed = settings["training_completed"]
    else:
        training_completed = []

    logging.info(f"Loading labels file flat_labels.parquet")
    y = utils.read_parquet_to_np(os.path.join(cleaned_data_dir, "flat_labels.parquet"))
    y = y.flatten()

    for finetuned_model in finetuned_models:
        hmodel, model, splitt, feature = finetuned_model.split("::")

        if hmodel != 'flat' and split == splitt:
            if finetuned_model in training_completed:
                logging.info(f"Skipping {finetuned_model} as it is already completed")
                continue

            features_path = os.path.join(features_dir, f"{feature}_features.parquet")
            X = utils.read_parquet_to_np(features_path)

            train_model_with_features(
                htype=hmodel,
                model=model,
                X=X,
                y=y,
                feature=feature,
                reports_dir=reports_dir,
                split=split,
                test_size=test_size,
            )
            training_completed.append(finetuned_model)
            settings["training_completed"] = training_completed
            utils.save_settings(settings, path="src_aurel/settings.json")
            logging.info(f"Trained models updated with {finetuned_model}")
