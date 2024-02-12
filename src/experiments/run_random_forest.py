from itertools import product
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

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

# Function to split labels into hierarchical levels
def split_labels(labels):
    return [label.split('.') for label in labels]

# Possible values for the parameters
ks = [6]
sizes = [64, 128]
strats = ["kmer", "chaos", "rtd", "spaced", "mash", "acs"]

# Initialize an empty DataFrame to store results
all_results = pd.DataFrame()

# Loop through all combinations of k, size, and strat
for k, size, strat in product(ks, sizes, strats):
    # Load the dataset based on the current combination of parameters
    if strat == "kmer":
        data = pd.read_parquet(f'../../data/features/{k}-mer_standard.parquet')
    elif strat == "chaos":
        data = pd.read_parquet(f'../../data/features/chaos_standard_{size}.parquet')
    elif strat == "rtd":
        data = pd.read_parquet(f'../../data/features/{k}-rtd.parquet')
    elif strat == "spaced":
        data = pd.read_parquet(f'../../data/features/{k}-spaced.parquet')
    elif strat == "mash":
        data = pd.read_parquet('../../data/features/mash.parquet')
    elif strat == "acs":
        data = pd.read_parquet('../../data/features/acs.parquet')

    # Encode the labels
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Target"])

    # Split the data into training and test sets
    X_train = data[data['Test'] == 0].drop(columns=["Target", "Test", "Label"])
    y_train = data[data['Test'] == 0]['Label']
    X_test = data[data['Test'] == 1].drop(columns=["Target", "Test", "Label"])
    y_test = data[data['Test'] == 1]['Label']

    # Train the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=15)
    rf.fit(X_train, y_train)

    # Predict the test set
    y_pred = rf.predict(X_test)

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
    experiment_id = f"{strat}_k{k}_size{size}"

    # Set the concatenated string as the index of the df_results DataFrame
    df_results.index = [experiment_id]

    # Append the results of this experiment to the all_results DataFrame
    all_results = pd.concat([all_results, df_results])

    all_results = all_results.round(4)
    all_results.to_csv("random_forest.csv")