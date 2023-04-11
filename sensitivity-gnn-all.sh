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
        directory="gnn-all-$i-trial-$trial"

        # Make the directory
        mkdir "$directory"

        # Copy the hyperparameters
        cp "gnn-search-all/best_hps.json" "$directory"

        echo "Running python module"

        # Run the Python module with the directory as an argument
        python -m camcann.lab GNNModel All -r "$ratio" -e 500 --and-uq --only-best "$directory"
    done
done
