# %%
import numpy as np
from collections import Counter
from itertools import product
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lib.feature_extraction.chaos_game import chaos_game_representation
import joblib
from Bio import SeqIO
import argparse
import os

def load_reference_sequences(args):
    fasta_files = [
        ("reference-1.fasta", 1),
        ("reference-2.fasta", 2),
        ("reference-3.fasta", 3),
        ("reference-4.fasta", 4)
    ]

    reference_folder = f'../references/'

    # Initialize an empty list to store the sequences and types
    data = []

    # Loop through the FASTA files and read the sequences
    for file_path, seq_type in fasta_files:
        for record in SeqIO.parse(f"{reference_folder}/{file_path}", "fasta"):
            data.append({"Sequence": str(record.seq), "Type": seq_type})

    # Create a DataFrame from the collected data
    refs_df = pd.DataFrame(data)

    # Initialize an empty list to store the new rows
    new_rows = []
    # Break sequences into sections
    section_length = 500
    overlap = 400

    for _, row in refs_df.iterrows():
        sequence = row["Sequence"]
        seq_type = row["Type"]
        seq_len = len(sequence)
        
        # Generate sections with the specified overlap
        start = 0
        while start + section_length <= seq_len:
            section = sequence[start : start + section_length]
            new_rows.append({"Sequence": section, "Type": seq_type, "Start": start})
            start += section_length - overlap

    # Create the new DataFrame
    ref_sections_df = pd.DataFrame(new_rows)
    ref_sections_df = ref_sections_df[ref_sections_df["Start"] <= 9500]

    refs_sections_chaos = chaos_game_representation(ref_sections_df, args.res, mode='BCGR')

    ref_df = pd.DataFrame(refs_sections_chaos)
    ref_df["Start"] = ref_sections_df["Start"]
    ref_df["Type"] = ref_sections_df["Type"]

    return ref_sections_df, refs_sections_chaos

def match_segments_to_reference(args, testing_sequences, ref_sections_df, refs_sections_chaos):
    middle_match = testing_sequences['Padded_Sequence'].str[100:-100]

    # Chaos game representation for the match sequences
    X_data = chaos_game_representation(pd.DataFrame({"Sequence": middle_match}), args.res, mode='BCGR')

    # Find initial match for Type and Position
    closest_rows = np.empty(X_data.shape[0], dtype=int)
    for i, x_row in enumerate(X_data):
        hamming_distances = np.sum(np.abs(refs_sections_chaos - x_row), axis=1)
        closest_rows[i] = np.argmin(hamming_distances)

    # Construct the match_df with matched reference sequences and start positions
    match_df = ref_sections_df.iloc[closest_rows].copy()
    match_df.reset_index(drop=True, inplace=True)

    # Add the matched reference sequence to the training_sequences DataFrame
    testing_sequences = testing_sequences.reset_index(drop=True)
    testing_sequences["Match_Sequence"] = match_df["Sequence"]
    testing_sequences["Match_Start"] = match_df["Start"]
    testing_sequences["Match Type"] = match_df["Type"]

    return testing_sequences

def optimize_shifts(args, testing_sequences):
    #Optimize Shifts
    optimized_sequences = []
    optimal_shifts = []

    for idx, row in testing_sequences.iterrows():
        # Extract the padded sequence and matched reference section
        padded_seq = row["Padded_Sequence"]  # Full segment
        matched_ref_seq = row["Match_Sequence"]  # Matched reference sequence

        # Generate all possible subsequences of length 500 from the padded sequence
        subsequences = [
            padded_seq[i : i + 500] for i in range(len(padded_seq) - 500 + 1)
        ]
        subsequences_df = pd.DataFrame({"Sequence": subsequences})

        # Calculate chaos game representations for all subsequences
        subsequences_chaos = chaos_game_representation(subsequences_df, args.res, mode='BCGR')

        # Calculate Hamming distances between the matched reference and all subsequences
        matched_ref_chaos = chaos_game_representation(
            pd.DataFrame({"Sequence": [matched_ref_seq]}), args.res, mode='BCGR'
        )

        # matched_ref_chaos = matched_ref_chaos[:, loaded_indices]
        hamming_distances = np.sum(np.abs(matched_ref_chaos - subsequences_chaos), axis=1)

        # Identify the subsequence with the minimum Hamming distance
        optimal_shift = np.argmin(hamming_distances)
        best_subsequence = subsequences[optimal_shift]

        # Append the optimized sequence and shift to the lists
        optimized_sequences.append(best_subsequence)
        optimal_shifts.append(optimal_shift)

    # Add the optimized sequences and shifts to the DataFrame
    testing_sequences["Optimal_Shift"] = optimal_shifts
    testing_sequences["Optimized_Sequence"] = optimized_sequences

    return testing_sequences

def load_and_predict(args, testing_sequences):
    # Initialize a list to store predictions
    predictions = []

    # Iterate over each row in the training_sequences DataFrame
    for idx, row in testing_sequences.iterrows():
        # Get the Match_Start position
        match_start = row["Match_Start"]
        match_type = row["Match Type"]

        model_filename = f'../models/{args.model_name}/denv_{match_type}/segment_{match_start}_rf.pkl'

        # Load the Random Forest model
        try:
            rf_model = joblib.load(model_filename)
        except FileNotFoundError:
            print(f"Model not found for Match_Start {match_start}. Skipping row {idx}.")
            predictions.append(None)
            continue

        # Prepare the feature vector for prediction
        optimized_seq = row["Optimized_Sequence"]
        chaos_representation = chaos_game_representation(
            pd.DataFrame({"Sequence": [optimized_seq]}), args.res, mode='BCGR'
        )

        # Make the prediction
        prediction = rf_model.predict(chaos_representation)[0]
        predictions.append(prediction)

        # Optional: Print progress
        print(f"Processed sequence {idx + 1}/{len(testing_sequences)}. Prediction: {prediction}")

    # Add the predictions to the DataFrame
    testing_sequences["Lineage"] = predictions

    # Output the results
    print("Classification predictions complete!")

        # Create predictions directory if it doesn't exist
    predictions_dir = '../predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save results to CSV
    output_path = f'{predictions_dir}/{args.dataset}_{args.run_name}_{args.model_name}_predictions.csv'
    testing_sequences["Accession ID"] = testing_sequences.groupby("Accession ID").cumcount().astype(str).radd(testing_sequences["Accession ID"] + "-")
    testing_sequences[["Accession ID", "Lineage"]].to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

def predict(args):
    ref_sections_df, refs_sections_chaos = load_reference_sequences(args)
    testing_sequences = pd.read_parquet(f"../../data/{args.dataset}.parquet", engine='pyarrow').rename(columns={"Sequence": "Padded_Sequence"})
    print(testing_sequences.head())
    testing_sequences = match_segments_to_reference(args, testing_sequences, ref_sections_df, refs_sections_chaos)
    testing_sequences = optimize_shifts(args, testing_sequences)

    # Ensure the Match_Start values are integers
    testing_sequences["Match_Start"] = testing_sequences["Match_Start"].astype(int)
    load_and_predict(args, testing_sequences)


def main():
    parser = argparse.ArgumentParser(description="Run a Random Forest on training data")
    parser.add_argument('--dataset', type=str, required=True, help='Viral Dataset name.')
    parser.add_argument('--run_name', type=str, required=True, help='Name for this run, used to create the output folder name.')
    parser.add_argument('--model_name', type=str, required=True, help='Name for the model, used in the output folder name.')
    parser.add_argument('--res', type=int, choices=[32, 64, 128, 256], default=128, help='Specifies the image resolution for FCGR feature extraction.')
    parser.add_argument('--criterion', choices=['gini', 'entropy'], default='entropy', help='Splitting Criterion')
    parser.add_argument('--class_weight', choices=['balanced', 'None'], default='balanced', help='Class Weight')
    
    # Parse arguments
    args = parser.parse_args()
    if args.class_weight == 'None':
        args.class_weight = None
    
    predict(args)

if __name__ == "__main__":
    main()