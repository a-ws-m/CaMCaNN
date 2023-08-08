#!/bin/bash

# Define the combinations of num_splits and num_repeats
splits=(2 3 4 5)
repeats=(3 2 1 1)

# Loop over each combination
for i in "${!splits[@]}"; do
    # Extract num_splits and num_repeats
    num_splits=${splits[i]}
    num_repeats=${repeats[i]}

    model_name="ecfp-nonionic-$num_splits-splits"

    echo "Running python module"

    # Run the python module with the arguments
    python -m camcann.lab --test-complementary --splits "$num_splits" --repeats "$num_repeats" ECFPLinear Nonionics "$model_name"
done
