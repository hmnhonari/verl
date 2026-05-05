#!/bin/bash

# Define your variations here
LOSS_MODES=("tvpo" "vanilla")
MODEL_MODES=("4B" "8B")

for model in "${MODEL_MODES[@]}"; do
   for loss in "${LOSS_MODES[@]}"; do
      sbatch verl/examples/cispo_trainer/sbatch_qwen3_8b_gsm8k.sh "$loss" "$model"
      sleep 1
   done
done