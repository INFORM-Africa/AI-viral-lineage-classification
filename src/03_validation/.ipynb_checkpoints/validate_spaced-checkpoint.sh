#!/bin/bash

# Directory containing the files you want to process
DIRECTORY="../../data/features/spaced"

# Loop through each file in the directory
for fullpath in "$DIRECTORY"/*
do
  # Extract the base name (file name without path)
  filename=$(basename -- "$fullpath")

  # Remove the file extension to get only the file name
  name="${filename%.*}"

  # Print the file name
  echo "Processing file: $name"

  # Run your Python script with the file name as an argument
  python scripts/validate_spaced.py "$name"
done