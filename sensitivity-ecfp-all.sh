#!/bin/bash

# Define the number of retrials to perform
num_trials=4

# Define a range for the loop
start=5
end=8

for trial in $(seq 1 $num_trials); do
    # Loop through the range
    for i in $(seq $start $end); do
        ratio="0.$i"

        # Define the directory name
        directory="ecfp-all-$i-trial-$trial"

        # Make the directory
        mkdir "$directory"

        echo "Running python module"

        # Run the Python module with the directory as an argument
        python -m camcann.lab ECFPLinear All -r "$ratio" --test-nist "$directory"
    done
done
