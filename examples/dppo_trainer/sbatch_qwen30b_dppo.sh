#!/bin/bash

#SBATCH --account=aip-gberseth
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h200:8
#SBATCH -o /scratch/h/homayoon/slurm-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=64
## #SBATCH --nodelist=tg11503

module load apptainer/1.4.5
module load httpproxy

export WANDB_API_KEY=$(cat $HOME/.wandb_key)
export WANDB_ENTITY=glen-berseth
export HYDRA_FULL_ERROR=1
# export LOSS_MODE=tvpo
export LOSS_MODE=${1:-tvpo}
export MODEL_MODE=${2:-4B}

# replace these information with your own
verl_workdir=/home/h/homayoon/verl
train_files=/home/h/homayoon/verl/data/dapo-math-17k.parquet
val_files=/home/h/homayoon/verl/data/aime-2024.parquet
apptainer_image_path=/scratch/h/homayoon/verl/verl.sif

# --- Start Ray Head Node ---
# On a single node, we just start the head. No need for IP detection or worker loops.
echo "Starting Ray head on $(hostname)"
srun --nodes=1 --ntasks=1 \
   apptainer run --nv --bind $verl_workdir $apptainer_image_path \
   ray start --head --port=6379 --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus=8 --block &

# Wait a moment for Ray to initialize
sleep 10

cd $HOME/verl

apptainer exec --nv \
    --bind /home/h/homayoon:/home/h/homayoon \
    --bind /scratch/h/homayoon:/scratch/h/homayoon \
    /scratch/h/homayoon/verl/verl.sif \
    bash /home/h/homayoon/verl/examples/dppo_trainer/run_qwen30b_dppo.sh
