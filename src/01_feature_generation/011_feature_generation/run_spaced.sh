#!/bin/bash

# Parameters for feature generation
data_type="SARS"

# Iterate over degenerate actions
for degenerate_action in remove replace; do

    # Loop for k=5, 6, 7
    for word_length in 5 6 7; do

        # Specify the validation results file name based on parameters
        validation_results_file="spaced_results/pattern_validation_${degenerate_action}_${word_length}_${data_type}.csv"

        # Iterate to generate, validate, and then delete the feature files
        for i in {1..3}; do  # Adjust the number of iterations as needed
            echo "Iteration $i for k=$word_length, degenerate=$degenerate_action"

            # Generate features and get the generated file path
            full_path=$(python scripts/feature_spaced.py --Word_Length $word_length --Degenerate $degenerate_action --Data $data_type | tail -n 1)

            # Extract the filename from the full path
            filename=$(basename "$full_path")

            # Use $filename to pass only the filename to the validation script
            python scripts/validate_spaced.py $filename -v $data_type -r $validation_results_file

            # Delete the generated file
            rm "$full_path"
        done

        # After all iterations, identify the best pattern from the specific validation results file
        best_pattern=$(head -2 $validation_results_file | tail -n 1 | cut -d',' -f2)

        # Regenerate the best feature set with the identified best pattern
        final_path=$(python scripts/feature_spaced.py --Word_Length $word_length --Degenerate $degenerate_action --Data $data_type -p $best_pattern | tail -n 1)

        # Rename the final file while keeping it in the original directory
        mv "$final_path" "$(dirname "$final_path")/spaced_${word_length}_${degenerate_action}.parquet"
        
    done
done

echo "Best features regenerated and stored with names spaced_{k}_{degenerate}.parquet"