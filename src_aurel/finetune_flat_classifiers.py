from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as metrics
import optuna
import pyarrow.parquet as pq
import utils, os
from datetime import datetime

def get_timestamp():
    now = datetime.now()
    return int(now.timestamp())

def finetune_rf(features, labels, features_name, test_size=0.2, n_trials=100):
    timestamp = get_timestamp()
    reports_filename = os.path.join(reports_dir, f'flat_rf_{features_name}_{timestamp}.txt')
    file = open(reports_filename, 'a')
    
    file.write(f"{features_name} features - {features.shape}\n")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    file.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}\n")
    
    file.write("Hyperparameters and search space\n")
    file.write("\tn_estimators: 50-500\n")
    file.write("\tmax_depth: 1-20\n")
    file.write("\tmin_samples_split: 2-20\n")
    file.write("\tmin_samples_leaf: 1-10\n")
    file.write("\tcriterion: gini, entropy\n")

    def objective(trial):
        # Define the search space for hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        
        # Instantiate the Random Forest Classifier with the suggested hyperparameters
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42,
            n_jobs=-1,
        )
    
        # Train the model
        rf.fit(X_train, y_train)
        
        # Calculate the accuracy score on the validation set
        y_pred = rf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        return accuracy
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.close()

    print(f"Finetuning completed with {study.best_value} accuracy")
    print("Logs saved to", reports_filename)


def finetune_xgb(features, labels, features_name, test_size=0.2, n_trials=100):
    timestamp = get_timestamp()
    reports_filename = os.path.join(reports_dir, f'flat_xgb_{features_name}_{timestamp}.txt')
    file = open(reports_filename, 'a')
    
    file.write(f"{features_name} features - {features.shape}\n")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    file.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}\n")
    
    file.write("Hyperparameters and search space\n")
    file.write("\tn_estimators: 50-500\n")
    file.write("\tmax_depth: 1-20\n")
    file.write("\tsubsample: 0.5-1.0\n")
    file.write("\tcolsample_bytree: 0.5-1.0\n")
    file.write("\tgamma: 1e-8-1.0\n")
    file.write("\tmin_child_weight: 1-10\n")

    def objective(trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        subsample = trial.suggest_uniform('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
        gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

        xgb = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            objective='multi:softmax',
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )

        # Fit the model on the training data
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

        y_pred = xgb.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        return accuracy
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.close()

    print(f"Finetuning completed with {study.best_value} accuracy")
    print("Logs saved to", reports_filename)
    
def finetune_lgbm(features, labels, features_name, test_size=0.2, n_trials=100):
    timestamp = get_timestamp()
    reports_filename = os.path.join(reports_dir, f'flat_lgbm_{features_name}_{timestamp}.txt')
    file = open(reports_filename, 'a')
    
    file.write(f"{features_name} features - {features.shape}\n")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    file.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}\n")
    
    file.write("Hyperparameters and search space\n")
    file.write("\tlearning_rate: 1e-3-0.1\n")
    file.write("\tn_estimators: 50-500\n")
    file.write("\tmax_depth: 1-20\n")
    file.write("\tsubsample: 0.5-1.0\n")
    file.write("\tcolsample_bytree: 0.5-1.0\n")
    file.write("\tmin_child_samples: 1-20\n")
    file.write("\tnum_leaves: 20-50\n")
    file.write("\tboosting_type: gbdt, dart, rf\n")

    def objective(trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        subsample = trial.suggest_uniform('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
        min_child_samples = trial.suggest_int('min_child_samples', 1, 20)
        num_leaves = trial.suggest_int('num_leaves', 20, 50)
        boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf'])

        # Initialize the LightGBM classifier with the hyperparameters
        lgbm = LGBMClassifier(
            boosting_type=boosting_type,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            num_leaves=num_leaves,
            objective='multiclass',
            random_state=42,
            n_jobs=-1,
        )

        # Fit the model on the training data
        lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

        # Calculate the validation accuracy
        y_pred = lgbm.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        return accuracy
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.close()

    print(f"Finetuning completed with {study.best_value} accuracy")
    print("Logs saved to", reports_filename)
    
def finetune_cb(features, labels, features_name, test_size=0.2, n_trials=100):
    timestamp = get_timestamp()
    reports_filename = os.path.join(reports_dir, f'flat_cb_{features_name}_{timestamp}.txt')
    file = open(reports_filename, 'a')
    
    file.write(f"{features_name} features - {features.shape}\n")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    file.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}\n")
    
    file.write("Hyperparameters and search space\n")
    file.write("\titerations: 100-1000\n")
    file.write("\tlearning_rate: 0.001-0.1\n")
    file.write("\tdepth: 4-10\n")
    file.write("\tl2_leaf_reg: 1e-2-10.0\n")

    def objective(trial):
        iterations = trial.suggest_int('iterations', 100, 1000)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        depth = trial.suggest_int('depth', 4, 10)
        l2_leaf_reg = trial.suggest_loguniform('l2_leaf_reg', 1e-2, 10.0)

        # Initialize the CatBoost classifier with the hyperparameters
        cb = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_state=42,
            verbose=0,
        )

        # Fit the model on the training data
        cb.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10, verbose=False)

        # Calculate the validation accuracy
        y_pred = cb.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        return accuracy
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.close()

    print(f"Finetuning completed with {study.best_value} accuracy")
    print("Logs saved to", reports_filename)
    

def finetune_nb(features, labels, features_name, test_size=0.2, n_trials=100):
    timestamp = get_timestamp()
    reports_filename = os.path.join(reports_dir, f'flat_nb_{features_name}_{timestamp}.txt')
    file = open(reports_filename, 'a')
    
    file.write(f"{features_name} features - {features.shape}\n")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    file.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}\n")
    
    file.write("Hyperparameters and search space\n")
    file.write("alpha: 0.01 - 1.0\n")

    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.01, 1.0)

        nb = GaussianNB(var_smoothing=alpha)
        
        # # Perform 10-fold cross-validation
        # scores = cross_val_score(nb, features, labels, cv=10, scoring='accuracy')
        # accuracy =  scores.mean()
    
        # Train the model
        nb.fit(X_train, y_train)
        
        # Calculate the accuracy score on the validation set
        y_pred = nb.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        return accuracy
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    file.write(f"Best hyperparameters: {str(study.best_params)}\n")
    file.write(f"Best accuracy: {str(study.best_value)}\n")
    file.close()

    print(f"Finetuning completed with {study.best_value} accuracy")
    print("Logs saved to", reports_filename)


if __name__ == "__main__":
    settings = utils.load_settings()
    features_dir = settings["features_dir"]
    cleaned_data_dir = settings["cleaned_data_dir"]
    reports_dir = settings["reports_dir"]
    trials = settings["optuna_trials"]
    
    # Load feature
    X_table = pq.read_table(os.path.join(features_dir, "murugaiah_features.parquet"))
    X_murugaiah = X_table.to_pandas().to_numpy()

    X_table = pq.read_table(os.path.join(features_dir, "kmer_features_6.parquet"))
    X_kmer_6 = X_table.to_pandas().to_numpy()

    X_table = pq.read_table(os.path.join(features_dir, "kmer_features_5.parquet"))
    X_kmer_5 = X_table.to_pandas().to_numpy()

    X_table = pq.read_table(os.path.join(features_dir, "fcgr_features.parquet"))
    X_fcgr = X_table.to_pandas().to_numpy()

    y_table = pq.read_table(os.path.join(cleaned_data_dir, "flat_labels.parquet"))
    y = y_table.to_pandas().to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y.flatten())

    features_dict = {
        "murugaiah": X_murugaiah,
        "kmer_6": X_kmer_6,
        "kmer_5": X_kmer_5,
        # "fcgr": X_fcgr
    }

    for name, features in features_dict.items():
        finetune_nb(features, y, name, n_trials=trials)
        finetune_rf(features, y, name, n_trials=trials)
        finetune_xgb(features, y, name, n_trials=trials)
        finetune_lgbm(features, y, name, n_trials=trials)
        finetune_cb(features, y, name, n_trials=trials)
