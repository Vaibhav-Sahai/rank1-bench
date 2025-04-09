#!/bin/bash

MODEL="Qwen/Qwen2.5-7B"
TASK="BrightRetrieval"
NUM_GPUS=2

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
    CUDA_VISIBLE_DEVICES=2,3 time bash launch_job.sh $MODEL $TASK $DATASET $NUM_GPUS
    echo "Completed job for dataset: $DATASET"
    echo "-------------------------------------"
done

echo "All jobs completed!"