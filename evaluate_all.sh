#!/bin/bash

# Define the path to the folder containing all config files
CONFIG_FOLDER="./all_models/"

# Define the path to the output file
OUTPUT_FILE="evaluation_results.txt"

# Clear the contents of the output file
> "$OUTPUT_FILE"

# Iterate over each config file in the folder
for config_file in "$CONFIG_FOLDER"/*/*.yaml; do
    # Run the python command
    output=$(python evaluate_models.py --config "$config_file")

    # Get the last line of the output
    last_line=$(echo "$output" | tail -n 1)

    # Append the last line to the output file
    echo "$last_line" >> "$OUTPUT_FILE"
done
