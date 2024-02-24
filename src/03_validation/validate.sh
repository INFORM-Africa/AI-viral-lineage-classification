#!/bin/bash

# Array of filenames without the '.parquet' extension
file_names=(
    # "5-mer_remove"
    # "6-mer_remove"
    # "7-mer_remove"
    # "ACS_remove"
    # "DSP_remove_justa"
    # "DSP_remove_pp"
    # "DSP_remove_real"
    # "DSP_replace_justa"
    # "DSP_replace_pp"
    # "DSP_replace_real"
    # "FCGR_remove_128"
    # "FCGR_remove_256"
    # "FCGR_remove_64"
    # "FCGR_replace_128"
    # "FCGR_replace_256"
    # "FCGR_replace_64"
    # "MASH_remove_13"
    # "MASH_remove_21"
    # "MASH_remove_8"
    # "RTD_5-mer_remove"
    # "RTD_6-mer_remove"
    # "DSP_remove_real_dist"
    # "DSP_replace_real_dist"
    "mash_distance"
)

# Path to your Random Forest Python script, adjust if necessary
rf_script_path="scripts/validate_random_forest.py"

# Iterate over the file names
for file_name in "${file_names[@]}"
do
    echo "Processing file: $file_name"
    # Execute the Random Forest Python script with the current file name as an argument
    python $rf_script_path $file_name

    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $file_name"
    else
        echo "Error processing: $file_name"
    fi
done
