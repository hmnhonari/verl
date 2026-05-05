#!/bin/bash

# Define your variations here
LOSS_MODES=("tvpo" "vanilla")
MODEL_MODES=("4B" "8B")

for model in "${MODEL_MODES[@]}"; do
   for loss in "${LOSS_MODES[@]}"; do
      sbatch verl/examples/dppo_trainer/sbatch_qwen30b_dppo.sh "$loss" "$model"
      sleep 1
   done
done