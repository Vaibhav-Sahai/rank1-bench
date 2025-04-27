#!/bin/bash
set -euo pipefail

# === CONFIGURE ===
model="unsloth/Qwen2.5-0.5B-Instruct"
dataset="BrightRetrieval"
# your full list of subtasks:
splits=(
  biology earth_science economics psychology robotics
  stackoverflow sustainable_living leetcode pony
  aops theoremqa_questions theoremqa_theorems
)
ngpus=4
ft_flag="--ft"   # remove or set to "" if you don't always want --ft
# =================

# build an array of space-separated splits for each GPU
declare -a gpu_splits
for i in "${!splits[@]}"; do
  gpu=$(( i % ngpus ))
  gpu_splits[$gpu]+=" ${splits[$i]}"
done

# for each GPU, launch a background worker that processes its queue
for gpu in $(seq 0 $((ngpus-1))); do
  (
    echo "▶ GPU $gpu queue: ${gpu_splits[$gpu]}"
    for split in ${gpu_splits[$gpu]}; do
      echo "[$(date +'%H:%M:%S')] GPU $gpu → running split: $split"
      # run your existing script; catch failures and log, but don't exit
      CUDA_VISIBLE_DEVICES=$gpu bash launch_job.sh \
        "$model" "$dataset" "$split" 1 $ft_flag \
        || echo "✗ split '$split' on GPU $gpu failed, moving on"
    done
    echo "✅ GPU $gpu queue complete"
  ) &
done

wait
echo "🎉 All splits done."

