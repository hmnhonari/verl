#!/bin/bash
# End-to-end RESUME test on an interactive node. Runs the ACTUAL run_qwen30b_dppo.sh
# which (via the pinned/overridable experiment_name) auto-resumes the latest global_step
# of a precision_aware 30B checkpoint. Confirms: (1) checkpoint load succeeds WITHOUT a
# host-RAM OOM-kill, (2) the first post-resume update_actor (optimizer step) passes,
# (3) training advances past the resume step.
# Override which checkpoint via RESUME_EXPERIMENT_NAME and where to log via OUTDIR.
module load apptainer/1.4.5 2>/dev/null
module load httpproxy 2>/dev/null

OUTDIR=${OUTDIR:-/scratch/h/homayoon/resume_test}
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
MEMLOG=$OUTDIR/mem.log
RUNLOG=$OUTDIR/run.log
RAYLOG=$OUTDIR/ray.log
echo "OUTDIR=$OUTDIR node=$(hostname) start=$(date) RESUME_EXPERIMENT_NAME=${RESUME_EXPERIMENT_NAME:-<script default>}"

export VERL_DISABLE_FULLY_PARALLEL_LOAD=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export RAY_DEDUP_LOGS=0        # isolate the load/OOM test from network; irrelevant to memory
export LOSS_MODE=tvpo
export MODEL_MODE=30B
export RESUME_EXPERIMENT_NAME=${RESUME_EXPERIMENT_NAME:-}
# Do NOT export CKPTS_DIR: the experiment_name in run_qwen30b_dppo.sh resolves it
# to the checkpoint dir so resume_mode=auto loads the latest global_step.

verl_workdir=/home/h/homayoon/verl
img=/scratch/h/homayoon/verl/verl.sif

# --- host + GPU memory monitor (3s) ---
(
  while true; do
    ts=$(date +%H:%M:%S)
    a=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
    echo "$ts MemAvailGB=$a GPUusedMiB=[$gpu]" >> "$MEMLOG"
    sleep 3
  done
) &
MONPID=$!

# --- Ray head (container, production-default object store) ---
apptainer run --nv --bind "$verl_workdir" "$img" \
   ray start --head --port=6379 --num-cpus 64 --num-gpus=8 --block > "$RAYLOG" 2>&1 &
sleep 15

# --- the ACTUAL training script (auto-resume) ---
cd "$HOME/verl"
apptainer exec --nv \
    --bind /home/h/homayoon:/home/h/homayoon \
    --bind /scratch/h/homayoon:/scratch/h/homayoon \
    "$img" \
    bash /home/h/homayoon/verl/examples/dppo_trainer/run_qwen30b_dppo.sh > "$RUNLOG" 2>&1
EXIT=$?
echo "RUN_EXIT=$EXIT end=$(date)" | tee -a "$RUNLOG"

kill "$MONPID" 2>/dev/null
apptainer run --nv "$img" ray stop > /dev/null 2>&1
echo "DONE exit=$EXIT outdir=$OUTDIR"
touch "$OUTDIR/DONE"
