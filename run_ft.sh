#!/bin/bash

MODEL="sumuks/relevance-reranker-tiny"
TASK="BrightRetrieval"
NUM_GPUS=1
EXTRA_FLAGS="--ft"

# List of datasets to iterate through
DATASETS=(
    # "economics"
    # "stackoverflow"
    # "sustainable_living"
    "leetcode"
    "pony"
    "aops"
    "theoremqa"
)

# Run the command for each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Running job for dataset: $DATASET"
    CUDA_VISIBLE_DEVICES=0 time bash launch_job.sh $MODEL $TASK $DATASET $NUM_GPUS $EXTRA_FLAGS
    echo "Completed job for dataset: $DATASET"
    echo "-------------------------------------"
done

echo "All jobs completed!"
