#!/bin/bash

# Loop for k=5, 6, 7
for k in 5 6 7; do
    # Run the Python script 25 times for each value of k
    for i in {1..50}; do
        echo "Iteration $i for k=$k"
        python feature-spaced.py --Degenerate Remove --Word_Length $k
    done
    
    for i in {1..50}; do
        echo "Iteration $i for k=$k"
        python feature-spaced.py --Degenerate Replace --Word_Length $k
    done
    
done