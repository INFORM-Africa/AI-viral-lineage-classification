from itertools import product
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Assuming calculate_f1_scores is a function that calculates and returns f1_scores_micro and f1_scores_macro
def calculate_f1_scores(y_test_decoded, y_pred_decoded):
    # Split the true and predicted decoded labels
    true_labels_split = split_labels(y_test_decoded)
    pred_labels_split = split_labels(y_pred_decoded)

    max_depth = max(max(len(label) for label in true_labels_split), max(len(label) for label in pred_labels_split))

    f1_scores_micro = []
    f1_scores_macro = []

    for level in range(max_depth):
        true_level_labels = [label[level] if level < len(label) else '' for label in true_labels_split]
        pred_level_labels = [label[level] if level < len(label) else '' for label in pred_labels_split]
        
        f1_micro = f1_score(true_level_labels, pred_level_labels, average='micro', zero_division=0)
        f1_macro = f1_score(true_level_labels, pred_level_labels, average='macro', zero_division=0)
        
        f1_scores_micro.append(f1_micro)
        f1_scores_macro.append(f1_macro)

    return max_depth, f1_scores_micro, f1_scores_macro

def norm(df):
    row_sums = df.sum(axis=1)
    # Normalize the DataFrame by dividing each row by its sum
    return df.div(row_sums, axis=0)

def run_experiment(data, all_results, name):
    # Encode the labels
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Target"])

    # Split the data into training and test sets
    X_train = data[data['Test'] == 0].drop(columns=["Target", "Test", "Label"])
    y_train = data[data['Test'] == 0]['Label']
    X_test = data[data['Test'] == 1].drop(columns=["Target", "Test", "Label"])
    y_test = data[data['Test'] == 1]['Label']
    
    X_train = norm(X_train)
    X_test = norm(X_test)

    xgb = XGBClassifier(n_estimators=100, random_state=42, verbosity=1, n_jobs=15, tree_method='gpu_hist')

    # Define your evaluation set(s)
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Train the model with the training set and watch the evaluation on eval_set
    xgb.fit(X_train, y_train, eval_set=eval_set, eval_metric="mlogloss", verbose=True)
    
    y_pred = xgb.predict(X_test)

    # Decode the labels
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Calculate F1 scores for each hierarchical level
    max_depth, f1_scores_micro, f1_scores_macro = calculate_f1_scores(y_test_decoded, y_pred_decoded)  # Assume this function is implemented similarly to your previous approach

    # Append global F1 scores
    f1_micro_global = f1_score(y_test, y_pred, average='micro')
    f1_macro_global = f1_score(y_test, y_pred, average='macro')
    f1_scores_micro.append(f1_micro_global)
    f1_scores_macro.append(f1_macro_global)

    # Compile results into a DataFrame
    column_names = [f"Level {i} F1 Score (Micro)" for i in range(1, max_depth + 1)] + ["Global F1 Score (Micro)"]
    column_names += [f"Level {i} F1 Score (Macro)" for i in range(1, max_depth + 1)] + ["Global F1 Score (Macro)"]
    df_results = pd.DataFrame([f1_scores_micro + f1_scores_macro], columns=column_names)

    # Concatenate the metadata into a single string to use as the index
    experiment_id = name

    # Set the concatenated string as the index of the df_results DataFrame
    df_results.index = [experiment_id]

    # Append the results of this experiment to the all_results DataFrame
    all_results = pd.concat([all_results, df_results])

    all_results = all_results.round(4)
    all_results.to_csv("xgboost_ffp.csv")
    return all_results

# Function to split labels into hierarchical levels
def split_labels(labels):
    return [label.split('.') for label in labels]

# Possible values for the parameters
ks = [5, 6]
sizes = [64, 128]
strats = ["kmer", "chaos", "rtd", "spaced", "mash", "acs"]

# Initialize an empty DataFrame to store results
all_results = pd.DataFrame()

# Loop through strategies
for strat in strats:
    if strat == "kmer":
        for k in ks:
            data = pd.read_parquet(f'../../data/features/{k}-mer_standard.parquet')
            all_results = run_experiment(data, all_results, f'{k}-ffp')
    # elif strat == "chaos":
    #     for size in sizes:
    #         data = pd.read_parquet(f'../../data/features/chaos_standard_{size}.parquet')
    #         all_results = run_experiment(data, all_results, f'chaos_standard_{size}')
    # elif strat in ["rtd", "spaced"]:
    #     for k in ks:
    #         data = pd.read_parquet(f'../../data/features/{k}-{strat}.parquet')
    #         all_results = run_experiment(data, all_results, f'{k}-{strat}')
    # elif strat in ["mash", "acs"]:
    #     data = pd.read_parquet(f'../../data/features/{strat}.parquet')
    #     all_results = run_experiment(data, all_results, strat)
