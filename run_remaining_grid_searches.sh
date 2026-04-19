#!/bin/bash

# Dataset list (already running industrial_and_scientific separately)
datasets=("video_games" "cell_phones_and_accessories")

for ds in "${datasets[@]}"
do
    echo "========================================="
    echo "Starting Grid Search for: $ds"
    echo "========================================="
    # Using -u for unbuffered output to log file
    /Users/rkaramyan/hybrid_recsys_capstone/.venv/bin/python -u src/grid_search.py --dataset "$ds" --num_samples 12 --epochs 50 >> grid_search_all_logs.txt 2>&1
    echo "Finished Grid Search for: $ds"
done
