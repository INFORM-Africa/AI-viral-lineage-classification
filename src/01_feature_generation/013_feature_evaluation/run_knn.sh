#!/bin/bash

data_type="SARS"

# Path to the CSV file
csv_file="parameters/${data_type}/knn_parameters.csv"

# Skip the header line and read the rest of the csv
tail -n +2 "$csv_file" | while IFS=, read -r feature k
do
    # Skip empty lines or lines without the required feature field
    if [ -z "$feature" ]; then
        continue
    fi

    # Start building the command
    cmd="python scripts/evaluate_knn.py '$feature' -v $data_type" 

    # Add n_estimators if it exists and is valid
    if [ -n "$k" ]; then
        cmd+=" --k $k"
    fi

    # Execute the command
    echo "Running: $cmd"
    eval $cmd
done
