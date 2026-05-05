#!/usr/bin/env bash
set -x

# Modes:
#   cispo      -> current CISPO-style settings
#   tvpo       -> match examples/dppo_trainer/run_qwen30b_dppo.sh
#   dppo_tv    -> match examples/dppo_trainer/run_qwen30b_dppo.sh
#   vanilla    -> match examples/dppo_trainer/run_qwen30b_dppo.sh
LOSS_MODE=${LOSS_MODE:-tvpo}

# gsm8k_train_path=${TRAIN_FILE:-/home/h/homayoon/verl/data/gsm8k/train.parquet}
# gsm8k_test_path=${TEST_FILE:-/home/h/homayoon/verl/data/gsm8k/test.parquet}
# math_train_path=${MATH_TRAIN_FILE:-/home/h/homayoon/verl/data/math/train.parquet}
# math_test_path=${MATH_TEST_FILE:-/home/h/homayoon/verl/data/math/test.parquet}

# train_files="['$gsm8k_train_path', '$math_train_path']"
# test_files="['$gsm8k_test_path', '$math_test_path']"

train_path=${TRAIN_FILE:-"/home/h/homayoon/verl/data/dapo-math-17k.parquet"}
test_path=${TEST_FILE:-"/home/h/homayoon/verl/data/aime-2024.parquet"}

train_files="['$train_path']"
test_files="['$test_path']"

MODEL_MODE=${MODEL_MODE:-8B}
MODEL_PATH=${MODEL_PATH:-/scratch/h/homayoon/verl/models/Qwen3-8B}
if [[ "$MODEL_MODE" == "8B" ]]; then
  MODEL_PATH=${MODEL_PATH:-/scratch/h/homayoon/verl/models/Qwen3-8B}
elif [[ "$MODEL_MODE" == "4B" ]]; then
  MODEL_PATH=${MODEL_PATH:-/scratch/h/homayoon/verl/models/Qwen3-4B-Base}
else
  echo "Invalid MODEL_MODE: ${MODEL_MODE}"
  echo "Expected one of: 8B, 4B"
  exit 1
fi

# Qwen3-8B-friendly defaults, based on examples/grpo_trainer/run_qwen3-8b.sh
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
# MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
# MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
# PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
# PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-10}

ROLLOUT_LOGPROB_MICRO_BATCH_SIZE=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-32}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
ROLLOUT_N=${ROLLOUT_N:-16}
n_resp_per_prompt_val=32

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}

ACTOR_LR=${ACTOR_LR:-1e-6}
USE_KL_IN_REWARD=False
CRITIC_WARMUP=0

# CISPO defaults from the current script; overridden for tvpo/dppo_tv/vanilla below.
USE_KL_LOSS=True
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0

EXTRA_ARGS=()

case "$LOSS_MODE" in
  cispo)
    CLIP_RATIO_LOW=${CLIP_LOW:-10}
    CLIP_RATIO_HIGH=${CLIP_HIGH:-0.2}
    ;;

  tvpo)
    # Preserved from examples/dppo_trainer/run_qwen30b_dppo.sh
    CLIP_RATIO=0.015
    CLIP_RATIO_LOW=${CLIP_LOW:-0.15}
    CLIP_RATIO_HIGH=${CLIP_HIGH:-0.005}

    USE_KL_LOSS=False
    EXTRA_ARGS+=(
    #   "algorithm.rollout_correction.bypass_mode=True"
    #   "algorithm.norm_adv_by_std_in_grpo=False"
      "actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}"
      "actor_rollout_ref.actor.clip_ratio_c=10000.0"
      "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm"
    #   "actor_rollout_ref.actor.calculate_entropy=True"
    )
    ;;

  dppo_tv)
    # Preserved from examples/dppo_trainer/run_qwen30b_dppo.sh
    CLIP_RATIO=0.15
    CLIP_RATIO_LOW=${CLIP_LOW:-0.15}
    CLIP_RATIO_HIGH=${CLIP_HIGH:-0.15}

    USE_KL_LOSS=False
    EXTRA_ARGS+=(
    #   "algorithm.rollout_correction.bypass_mode=True"
    #   "algorithm.norm_adv_by_std_in_grpo=False"
      "actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}"
      "actor_rollout_ref.actor.clip_ratio_c=10000.0"
      "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm"
    #   "actor_rollout_ref.actor.calculate_entropy=True"
    )
    ;;

  vanilla)
    # Preserved from examples/dppo_trainer/run_qwen30b_dppo.sh
    CLIP_RATIO=0.2
    CLIP_RATIO_LOW=${CLIP_LOW:-0.2}
    CLIP_RATIO_HIGH=${CLIP_HIGH:-0.2}

    USE_KL_LOSS=False
    EXTRA_ARGS+=(
    #   "algorithm.rollout_correction.bypass_mode=True"
    #   "algorithm.norm_adv_by_std_in_grpo=False"
      "actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}"
      "actor_rollout_ref.actor.clip_ratio_c=10000.0"
      "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm"
    #   "actor_rollout_ref.actor.calculate_entropy=True"
    )
    ;;

  *)
    echo "Invalid LOSS_MODE: ${LOSS_MODE}"
    echo "Expected one of: cispo, tvpo, dppo_tv, vanilla"
    exit 1
    ;;
esac

# REWARD_CONFIG="
#     reward.reward_manager.name=dapo \
#     +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
#     +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
#     +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
#     +reward.reward_kwargs.overlong_buffer_cfg.log=False \
#     +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}"

current_seconds=$(date +%s)
PROJECT_NAME=${PROJECT_NAME:-verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_MODE}_${LOSS_MODE}_clip${CLIP_RATIO}_${current_seconds}}

CKPTS_DIR=${CKPTS_DIR:-"/scratch/h/homayoon/verl/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "${EXTRA_ARGS[@]}" \
    # $REWARD_CONFIG \
    # "${REWARD_CONFIG[@]}" \
    "$@"


# $REWARD_CONFIG \
    