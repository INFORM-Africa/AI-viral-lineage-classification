#!/bin/bash

# Array of filenames without the '.parquet' extension
file_names=(
    #"5-mer_remove"
    #"DSP_remove_real"
    "FCGR_replace_128"
    "MASH_remove_13"
    "RTD_5-mer_replace"
    "5-mer_replace"
    "DSP_replace_justa_dist"
    #"FCGR_replace_256"
    "MASH_remove_21"
    "RTD_6-mer_remove"
    "6-mer_remove"
    "DSP_replace_pp_dist"
    "FCGR_replace_64"
    "MASH_remove_8"
    "RTD_6-mer_replace"
    "6-mer_replace"
    "DSP_replace_real_dist"
    "MASH_distance_remove_13"
    "MASH_replace_13"
    "RTD_7-mer_remove"
    "7-mer_remove"
    "FCGR_remove_128"
    "MASH_distance_remove_21"
    "MASH_replace_21"
    "RTD_7-mer_replace"
    "7-mer_replace"
    #"FCGR_remove_256"
    "MASH_distance_remove_8"
    "MASH_replace_8"
    "DSP_remove_real_dist"
    "FCGR_remove_64"
    "MASH_distance_replace_8"
    "RTD_5-mer_remove"
)

# Path to your Random Forest Python script, adjust if necessary
rf_script_path="scripts/validate_xgboost.py"

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
