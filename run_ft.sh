#!/bin/bash

MODEL="../benchmark/qwen2.5-7b-reasoning-merged/"
TASK="BrightRetrieval"
NUM_GPUS=2
EXTRA_FLAGS="--ft"

# List of datasets to iterate through
DATASETS=(
    "biology"
    "earth_science"
    "economics"
    "psychology"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "leetcode"
    "pony"
    "aops"
    "theoremqa"
)

# Run the command for each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Running job for dataset: $DATASET"
    CUDA_VISIBLE_DEVICES=0,1 time bash launch_job.sh $MODEL $TASK $DATASET $NUM_GPUS $EXTRA_FLAGS
    echo "Completed job for dataset: $DATASET"
    echo "-------------------------------------"
done

echo "All jobs completed!"