#!/bin/bash

# Submit a batch of runs. Each call: sbatch_qwen30b_dppo.sh <loss> <model> <clip>
# The clip arg is only read by the tvpo loss mode; vanilla and dppo_tv hardcode
# their own clip in run_qwen30b_dppo.sh, so the value passed for those is cosmetic
# (kept matching for readable experiment-dir names).
SBATCH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/sbatch_qwen30b_dppo.sh"

# --- 4B + 8B: tvpo (clip=0.02) and grpo (vanilla, defaults) ---
for model in "4B" "8B"; do
   sbatch "$SBATCH" tvpo    "$model" 0.02   # tvpo, clip=0.02
   sleep 1
   sbatch "$SBATCH" vanilla "$model" 0.2    # grpo baseline, default clip
   sleep 1
done

# --- 30B: dppo (TV divergence) ---
sbatch "$SBATCH" dppo_tv 30B 0.15
sleep 1
