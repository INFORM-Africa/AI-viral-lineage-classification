# %%
import numpy as np
from collections import Counter
from itertools import product
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lib.feature_extraction.chaos_game import chaos_game_representation
import joblib
import argparse
from Bio import SeqIO
import os

def train(args):
    # Iterate through segments
    serotypes = ['1', '2', '3', '4']

    for type in serotypes:
        for start in range(100, 9500, 100):
            segment_file = f"../data/{args.dataset}/segment_{start}.csv"
            
            # Check if the segment file exists
            if not os.path.exists(segment_file):
                print(f"Segment file {segment_file} not found. Skipping.")
                continue

            # Load the segment data
            training_segment = pd.read_csv(segment_file).rename(columns={"Segment_Sequence": "Sequence"})
            training_segment = training_segment.astype({"Clade": str})
            training_segment = training_segment[training_segment["Clade"].str[0] == type]
            X_train = chaos_game_representation(training_segment, args.res, mode='BCGR')

            # Define targets
            y_train = training_segment['Clade']

            # Train the Random Forest model
            rf = RandomForestClassifier(random_state=42, criterion=args.criterion, class_weight=args.class_weight)
            rf.fit(X_train, y_train)

            # Save the trained model
            model_dir = f'../../models/{args.model_name}/denv_{type}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/segment_{start}_rf.pkl"
            joblib.dump(rf, model_path, compress=6)
            print(f"Random Forest for segment {start} saved to {model_path}.")

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
    
    train(args)

if __name__ == "__main__":
    main()
