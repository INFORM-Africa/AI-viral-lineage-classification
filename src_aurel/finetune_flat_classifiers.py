import logging, optuna, utils, os, time
from model_selection import cross_val_score, train_test_split_score
from sklearn import metrics as metrics
from datetime import datetime

# Import the classifiers
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

h_type = "flat"


def get_timestamp():
    now = datetime.now()
    return int(now.timestamp())

def finetune_rf(features, labels, features_name, reports_dir, n_trials, cv, test_size=None):
    timestamp = get_timestamp()
    split = "cv" if cv else "tts"
    reports_filename = os.path.join(reports_dir, f'{h_type}_rf_{split}_{features_name}_{timestamp}.txt')

    if not cv:
        assert test_size is not None, "test_size must be provided for train-test-split"
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42,
            n_jobs=-1,
        )

        if cv:
            scores = cross_val_score(rf, features, labels, n_splits=5)
            accuracy = scores.mean()
        else:
            accuracy = train_test_split_score(rf, features, labels, test_size=test_size)

        return accuracy
    
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    end_time = time.time()
    duration = end_time - start_time

    file = open(reports_filename, 'a')
    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.write(f"Execution time: {duration:.4f} seconds\n")
    file.close()

    logging.info(f"Finetuning completed with {study.best_value} accuracy")
    logging.info(f"Logs saved to {reports_filename}")


def finetune_xgb(features, labels, features_name, reports_dir, n_trials, cv, test_size=None):
    timestamp = get_timestamp()
    split = "cv" if cv else "tts"
    reports_filename = os.path.join(reports_dir, f'flat_xgb_{split}_{features_name}_{timestamp}.txt')

    if not cv:
        assert test_size is not None, "test_size must be provided for train-test-split"
    
    def objective(trial):
        gamma = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        
        xgb = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            gamma=gamma,
            min_child_weight=min_child_weight,
            objective='multi:softmax',
            random_state=42,
            verbosity=3,
            device = "cuda",
            eval_metric='mlogloss',
        )

        if cv:
            scores = cross_val_score(xgb, features, labels, n_splits=5)
            accuracy = scores.mean()
        else:
            accuracy = train_test_split_score(xgb, features, labels, test_size=test_size)

        return accuracy
    
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    end_time = time.time()
    duration = end_time - start_time

    file = open(reports_filename, 'a')
    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.write(f"Execution time: {duration:.4f} seconds\n")
    file.close()

    logging.info(f"Finetuning completed with {study.best_value} accuracy")
    logging.info(f"Logs saved to {reports_filename}")
    
def finetune_lgbm(features, labels, features_name, reports_dir, n_trials, cv, test_size=None):
    timestamp = get_timestamp()
    split = "cv" if cv else "tts"
    reports_filename = os.path.join(reports_dir, f'flat_lgbm_{split}_{features_name}_{timestamp}.txt') 

    if not cv:
        assert test_size is not None, "test_size must be provided for train-test-split"   
    
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_child_samples = trial.suggest_int('min_child_samples', 1, 10)
        num_iteration = trial.suggest_int('num_iteration', 50, 1000)
        num_leaves = trial.suggest_int('num_leaves', 50, 100)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)

        lgbm = LGBMClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            num_leaves=num_leaves,
            num_iteration=num_iteration,
            min_child_samples=min_child_samples,
            objective='multiclass',
            random_state=42,
        )

        if cv:
            scores = cross_val_score(lgbm, features, labels, n_splits=5)
            accuracy = scores.mean()
        else:
            accuracy = train_test_split_score(lgbm, features, labels, test_size=test_size)

        return accuracy
    
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    end_time = time.time()
    duration = end_time - start_time

    file = open(reports_filename, 'a')
    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.write(f"Execution time: {duration:.4f} seconds\n")
    file.close()

    logging.info(f"Finetuning completed with {study.best_value} accuracy")
    logging.info(f"Logs saved to {reports_filename}")
    
def finetune_cb(features, labels, features_name, reports_dir, n_trials, cv, test_size=None):
    timestamp = get_timestamp()
    split = "cv" if cv else "tts"
    reports_filename = os.path.join(reports_dir, f'flat_cb_{split}_{features_name}_{timestamp}.txt')

    if not cv:
        assert test_size is not None, "test_size must be provided for train-test-split"
    
    def objective(trial):
        bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1.0)
        depth = trial.suggest_int('depth', 1, 10)
        iterations = trial.suggest_int('iterations', 50, 1000)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
        random_strength = trial.suggest_float('random_strength', 0, 1.0)

        cb = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
            random_state=42,
            verbose=0,
            task_type="GPU",
            objective="RMSE",
        )

        if cv:
            scores = cross_val_score(cb, features, labels, n_splits=5)
            accuracy = scores.mean()
        else:
            accuracy = train_test_split_score(cb, features, labels, test_size=test_size)

        return accuracy
    
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    end_time = time.time()
    duration = end_time - start_time

    file = open(reports_filename, 'a')
    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.write(f"Execution time: {duration:.4f} seconds\n")
    file.close()

    logging.info(f"Finetuning completed with {study.best_value} accuracy")
    logging.info(f"Logs saved to {reports_filename}")


def finetune_model_with_features(model:str, X, y, feature:str, n_trials, reports_dir:str, cv:bool, test_size:float=None):
    if model == 'rf':
        logging.info(f"Finetuning {h_type.upper()} Random Forest with {feature} features")
        finetune_rf(X, y, feature, reports_dir, n_trials=n_trials["rf"], cv=cv, test_size=test_size)
    elif model == 'xgb':
        logging.info(f"Finetuning {h_type.upper()} XGBoost with {feature} features")
        finetune_xgb(X, y, feature, reports_dir, n_trials=n_trials["xgb"], cv=cv, test_size=test_size)
    elif model == 'lgbm':
        logging.info(f"Finetuning {h_type.upper()} LightGBM with {feature} features")
        finetune_lgbm(X, y, feature, reports_dir, n_trials=n_trials["lgbm"], cv=cv, test_size=test_size)
    elif model == 'cb':
        logging.info(f"Finetuning {h_type.upper()} CatBoost with {feature} features")
        finetune_cb(X, y, feature, reports_dir, n_trials=n_trials["cb"], cv=cv, test_size=test_size)
    else:
        raise ValueError(f"Invalid model: {model}")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    settings = utils.load_settings(path="src_aurel/settings.json")
    params = utils.load_feature_params(path="src_aurel/feature_params.json")

    logging.info("Loading settings")
    features_dir = settings["features_dir"]
    cleaned_data_dir = settings["cleaned_data_dir"]
    reports_dir = settings["reports_dir"]
    n_trials = settings["optuna_trials"]
    fine_tune_models = settings["fine_tune_models"]
    test_size = settings["test_size"]
    cv = settings["use_cross_validation"]
    split = "cv" if cv else "tts"

    os.makedirs(reports_dir, exist_ok=True)

    if "fine_tune_completed" in settings:
        fine_tune_completed = settings["fine_tune_completed"]
    else:
        fine_tune_completed = []

    logging.info(f"Loading labels file flat_labels.parquet")
    y = utils.read_parquet_to_np(os.path.join(cleaned_data_dir, "flat_labels.parquet"))
    y = y.flatten()

    for model in fine_tune_models:
        if model not in ['rf', 'xgb', 'lgbm', 'cb']:
            raise ValueError(f"Invalid model: {model}")
    
        for key, config in params.items():
            if key == 'kmer':
                ks = config['k']
                for k in ks:
                    fine_tune_key = f"{h_type}::{model}::{split}::kmer_{k}"
                    if fine_tune_key not in fine_tune_completed:
                        logging.info(f"Loading features file kmer_{k}_features.parquet")
                        features_path = os.path.join(features_dir, f"kmer_{k}_features.parquet")
                        X = utils.read_parquet_to_np(features_path)
                        finetune_model_with_features(
                            model=model, 
                            X=X, 
                            y=y, 
                            feature=f"kmer_{k}", 
                            n_trials=n_trials, 
                            reports_dir=reports_dir,
                            cv=cv,
                            test_size=test_size,
                        )
                        fine_tune_completed.append(fine_tune_key)
                        settings["fine_tune_completed"] = fine_tune_completed
                        utils.save_settings(settings, path="src_aurel/settings.json")
                        logging.info(f"Fine tuned models updated with {fine_tune_key}")
                    else:
                        logging.info(f"Skipping {fine_tune_key} as it is already completed")
                    
            elif key == 'fcgr':
                ress = config['resolution']
                for res in ress:
                    fine_tune_key = f"{h_type}::{model}::{split}::fcgr_{res}"
                    if fine_tune_key not in fine_tune_completed:
                        logging.info(f"Loading features file fcgr_{res}_features.parquet")
                        features_path = os.path.join(features_dir, f"fcgr_{res}_features.parquet")
                        X = utils.read_parquet_to_np(features_path)
                        finetune_model_with_features(
                            model=model, 
                            X=X, 
                            y=y, 
                            feature=f"fcgr_{res}", 
                            n_trials=n_trials, 
                            reports_dir=reports_dir,
                            cv=cv,
                            test_size=test_size,
                        )

                        fine_tune_completed.append(fine_tune_key)
                        settings["fine_tune_completed"] = fine_tune_completed
                        utils.save_settings(settings, path="src_aurel/settings.json")
                        logging.info(f"Fine tuned models updated with {fine_tune_key}")
                    else:
                        logging.info(f"Skipping {fine_tune_key} as it is already completed")
                    
            elif key == 'murugaiah':
                fine_tune_key = f"{h_type}::{model}::{split}::murugaiah"
                if fine_tune_key not in fine_tune_completed:
                    logging.info(f"Loading features file murugaiah_features.parquet")
                    features_path = os.path.join(features_dir, f"murugaiah_features.parquet")
                    X = utils.read_parquet_to_np(features_path)
                    finetune_model_with_features(
                        model=model, 
                        X=X,
                        y=y, 
                        feature=f"murugaiah", 
                        n_trials=n_trials, 
                        reports_dir=reports_dir,
                        cv=cv,
                        test_size=test_size,
                    )
                    fine_tune_completed.append(fine_tune_key)
                    settings["fine_tune_completed"] = fine_tune_completed
                    utils.save_settings(settings, path="src_aurel/settings.json")
                    logging.info(f"Fine tuned models updated with {fine_tune_key}")
                else:
                    logging.info(f"Skipping {fine_tune_key} as it is already completed")
                
            else:
                raise ValueError(f"Invalid key: {key}")


