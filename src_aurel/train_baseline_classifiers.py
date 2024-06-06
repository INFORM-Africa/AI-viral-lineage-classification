from sklearn.dummy import DummyClassifier
from model_selection import train_test_split_predict
import time, os, logging, utils

h_type = 'baseline'

def train_model(strategy, X, y, test_size, split, reports_dir):
    start_time = time.time()
    reports_filename = os.path.join(reports_dir, f'training_{h_type}_{strategy}_{split}.txt')
    predictions_path = os.path.join(reports_dir, f'{h_type}_{strategy}_{split}_predictions.parquet')

    clf = DummyClassifier(strategy=strategy)

    accuracy, mcc, report = train_test_split_predict(
        estimator=clf, 
        X=X, 
        y=y, 
        test_size=test_size,
        path=predictions_path,
    )
    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} with Accuracy : {accuracy}.\nSaving logs ...\n")

    # Save logs
    with open(reports_filename, 'a') as file:
        file.write(report)
        file.write(f"\nMCC: {mcc}\n")
        file.write(f"\Accuracy: {accuracy}\n")

    utils.write_training_duration(
        h_model=h_type, 
        model=strategy,
        feature="",
        reports_dir=reports_dir,
        duration=elapsed_time
    )
    
    logging.info(f"Training completed in {elapsed_time:.2f} with \n{report}")
    logging.info(f"Logs saved to {reports_filename}")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    settings = utils.load_settings(path="src_aurel/settings.json")

    logging.info("Loading settings")
    features_dir = settings["features_dir"]
    cleaned_data_dir = settings["cleaned_data_dir"]
    reports_dir = settings["reports_dir"]
    test_size = settings["test_size"]
    baseline_classifiers = settings["baseline_classifiers"]
    cv = settings["use_cross_validation"]
    split = "cv" if cv else "tts"

    if "training_completed" in settings:
        training_completed = settings["training_completed"]
    else:
        training_completed = []

    logging.info(f"Loading labels file flat_labels.parquet")
    y = utils.read_parquet_to_np(os.path.join(cleaned_data_dir, "flat_labels.parquet"))
    y = y.flatten()

    logging.info(f"Loading features file murugaiah_features.parquet")
    features_path = os.path.join(features_dir, f"murugaiah_features.parquet")
    X = utils.read_parquet_to_np(features_path)

    for baseline_classifier in baseline_classifiers:
        training_key = f"{h_type}::{baseline_classifier}::{split}"
        if training_key in training_completed:
            logging.info(f"Skipping {training_key} as it is already completed")
            continue

        train_model(
            strategy=baseline_classifier,
            X=X,
            y=y,
            test_size=test_size,
            split=split, 
            reports_dir=reports_dir,
        )
        training_completed.append(training_key)
        settings["training_completed"] = training_completed
        utils.save_settings(settings, path="src_aurel/settings.json")
        logging.info(f"Trained models updated with {training_key}")

