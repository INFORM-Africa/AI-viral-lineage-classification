#!/bin/bash
data_type="SARS"
model="random_forest"

# Path to the directory containing the .parquet files
directory_path="../../../data/features/${data_type}"

# Path to your Random Forest Python script, adjust if necessary
rf_script_path="scripts/validate_${model}.py"

# Iterate over .parquet files in the given directory
for file_path in "$directory_path"/*.parquet
do
    # Extract file name without the directory path and extension
    file_name=$(basename "$file_path" .parquet)

    echo "Processing file: $file_name"
    # Execute the Random Forest Python script with the current file name as an argument
    python $rf_script_path $file_name -v "$data_type"

    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $file_name"
    else
        echo "Error processing: $file_name"
    fi
done
