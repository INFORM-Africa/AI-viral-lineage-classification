import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Random Forest Grid Search with Command Line Arguments")
    parser.add_argument('-m', '--Model', choices=['random_forest', 'xgboost', 'knn'], required=True, help='Specify the classification model')
    parser.add_argument('-v', '--Data', choices=['SARS', 'HIV'], required=True, help='Specify the virus dataset.')
    args = parser.parse_args()
    
    # Step 1: Read the CSV
    df = pd.read_csv(f'results/{args.Data}/{args.Model}_tuning_results.csv')  # Replace 'your_file.csv' with the actual file name

    # Step 2: Find the Best Model for Each Feature Type
    best_models = df.loc[df.groupby('features')['accuracy'].idxmax()]
    best_models['grouping_feature'] = best_models['features'].str.replace("_replace", "").str.replace("_remove", "")

    # Step 2: Filter the dataframe based on the highest accuracy within each group.
    best_models = best_models.sort_values('accuracy', ascending=False).drop_duplicates('grouping_feature').drop(columns=['grouping_feature']).sort_index()

    # Assuming best_models is a DataFrame and args.Data & args.Model are predefined variables
    directory = os.path.join("..", "013_feature_evaluation", "parameters", args.Data)
    csv_file_path = os.path.join(directory, f"{args.Model}_parameters.csv")

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save the specified columns of the DataFrame to CSV
    best_models[["features", "max_depth", "criterion", "class_weight"]].to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()