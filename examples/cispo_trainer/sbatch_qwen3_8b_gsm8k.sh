#!/bin/bash

#SBATCH --account=aip-gberseth
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH -o /scratch/h/homayoon/slurm-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --cpus-per-task=48

module load apptainer/1.4.5
module load httpproxy

export WANDB_API_KEY=$(cat $HOME/.wandb_key)
export WANDB_ENTITY=glen-berseth
export HYDRA_FULL_ERROR=1
export LOSS_MODE=tvpo

# replace these information with your own
verl_workdir=/home/h/homayoon/verl
train_files=/home/h/homayoon/verl/data/gsm8k/train.parquet
val_files=/home/h/homayoon/verl/data/gsm8k/test.parquet
apptainer_image_path=/scratch/h/homayoon/verl/verl.sif

# --- Start Ray Head Node ---
# On a single node, we just start the head. No need for IP detection or worker loops.
echo "Starting Ray head on $(hostname)"
srun --nodes=2 --ntasks=1 \
   apptainer run --nv --bind $verl_workdir $apptainer_image_path \
   ray start --head --port=6379 --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus=4 --block &

# Wait a moment for Ray to initialize
sleep 10

cd $HOME/verl

apptainer exec --nv \
    --bind /home/h/homayoon:/home/h/homayoon \
    --bind /scratch/h/homayoon:/scratch/h/homayoon \
    /scratch/h/homayoon/verl/verl.sif \
    bash /home/h/homayoon/verl/examples/cispo_trainer/run_cispo_qwen3_8b_gsm8k.sh
