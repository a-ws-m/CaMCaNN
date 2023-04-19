#!/bin/bash

# Define the combinations of num_splits and num_repeats
splits=(2 3 4 5)
repeats=(3 2 1 1)

# Loop over each combination
for i in "${!splits[@]}"; do
    # Extract num_splits and num_repeats
    num_splits=${splits[i]}
    num_repeats=${repeats[i]}

    base_dir_name="gnn-nonionics-$num_splits-splits"

    # Loop over each trial
    for trial in $(seq 0 $((num_repeats * num_splits - 1))); do
        # Create the directory
        dir_name="$base_dir_name-trial-$trial"
        mkdir "models/$dir_name"

        # Copy the hyperparameters
        cp "gnn-search-nonionics/best_hps.json" "$dir_name"
    done

    echo "Running python module"

    # Run the python module with the arguments
    python -m camcann.lab -e 500 --and-uq --only-best --splits "$num_splits" --repeats "$num_repeats" GNNModel Nonionics "$base_dir_name"
done
