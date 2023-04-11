#!/bin/bash

# Define the combinations of num_splits and num_repeats
splits=(2 3 4 5)
repeats=(3 2 1 1)

# Loop over each combination
for i in "${!splits[@]}"; do
    # Extract num_splits and num_repeats
    num_splits=${splits[i]}
    num_repeats=${repeats[i]}

    # Create the directory
    dir_name="ecfp-all-$num_splits-splits"

    echo "Running python module"

    # Run the python module with the arguments
    python -m camcann.lab --test-nist --splits "$num_splits" --repeats "$num_repeats" ECFPLinear All "$dir_name"
done
