#!/bin/bash

# Arrays of different arguments to iterate over
degenerates=("remove" "replace")
viruses=("SARS")
k_values_for_mash=("21" "23")
k_values_other=("6" "7")
r_values=("128" "256")
n_values=("Real" "PP" "JustA")

# Execute scripts with varying parameters
for degenerate in "${degenerates[@]}"; do
    for virus in "${viruses[@]}"; do
        python scripts/feature_acs.py -d "$degenerate" -v "$virus"
        
        for r_value in "${r_values[@]}"; do
            python scripts/feature_chaos.py -d "$degenerate" -v "$virus" -r "$r_value"
        done
        
        for n_value in "${n_values[@]}"; do
            python scripts/feature_gsp.py -d "$degenerate" -n "$n_value" -v "$virus"
        done
        
        for k_value in "${k_values_for_mash[@]}"; do
            python scripts/feature_mash.py -d "$degenerate" -k "$k_value" -v "$virus"
        done

        for k_value in "${k_values_other[@]}"; do
            python scripts/feature_kmer.py -d "$degenerate" -k "$k_value" -v "$virus"
            python scripts/feature_ffp.py -f "${k_value}-mer_${degenerate}" -v "$virus"
            python scripts/feature_rtd.py -d "$degenerate" -v "$virus" -k "$k_value"
        done
    done
done

echo "All feature scripts executed."