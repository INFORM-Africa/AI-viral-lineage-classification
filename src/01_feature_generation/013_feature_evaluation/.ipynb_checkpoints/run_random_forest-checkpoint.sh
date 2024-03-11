#!/bin/bash

data_type="SARS"

# Path to the CSV file
csv_file="parameters/${data_type}/random_forest_parameters.csv"

# Skip the header line and read the rest of the csv
tail -n +2 "$csv_file" | while IFS=, read -r feature max_depth criterion class_weight
do
    # Skip empty lines or lines without the required feature field
    if [ -z "$feature" ]; then
        continue
    fi

    # Start building the command
    cmd="python scripts/evaluate_random_forest.py '$feature' -v '$data_type'"

    # Add max_depth if it exists and is a number
    if [[ "$max_depth" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        cmd+=" --max_depth ${max_depth%.*}"  # Convert to integer if necessary
    fi

    # Add criterion if it exists and is valid
    if [ -n "$criterion" ]; then
        cmd+=" --criterion $criterion"
    fi

    # Add class_weight if it exists
    if [ -n "$class_weight" ]; then
        cmd+=" --class_weight $class_weight"
    fi

    # Execute the command
    echo "Running: $cmd"
    eval $cmd
done
