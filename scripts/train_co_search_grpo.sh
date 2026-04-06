#!/bin/bash
#SBATCH --job-name=cosearch-grpo
#SBATCH --chdir=/work/hzeng_umass_edu/ir-research/CoSearch
#SBATCH --output=/work/hzeng_umass_edu/ir-research/CoSearch/sbatch_out/cosearch_grpo_%j.out
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --constraint=vram80,bf16
#SBATCH --time=7-00:00:00

set -euo pipefail

# =========================
# Conda environment
# =========================
. /work/hzeng_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate search-llm

# =========================
# Resolve project paths
# (replaces Docker /workspace/ paths)
# =========================
PROJECT_ROOT=/work/hzeng_umass_edu/ir-research/CoSearch

echo "Project root: ${PROJECT_ROOT}"

# Sanity checks
if [ ! -f "${PROJECT_ROOT}/main_co_search_ppo.py" ]; then
    echo "ERROR: cannot find main_co_search_ppo.py under ${PROJECT_ROOT}."
    exit 1
fi

if [ ! -d "${PROJECT_ROOT}/verl" ]; then
    echo "ERROR: verl source dir not found at ${PROJECT_ROOT}/verl"
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/config/co_search_agent_loop_config.yaml" ]; then
    echo "ERROR: agent loop config not found at ${PROJECT_ROOT}/config/"
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/checkpoints" \
         "${PROJECT_ROOT}/data" \
         "${PROJECT_ROOT}/sbatch_out"

# =========================
# Python path
# verl is not pip-installed — add it so `import verl` resolves correctly
# =========================
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}"

# =========================
# vLLM backend config
# =========================
export VLLM_DISABLE_FLASHINFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# =========================
# Network interface (adjust to your cluster's infiniband/ethernet interface)
# =========================
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-en0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-en0}"

# =========================
# HuggingFace cache
# =========================
export HF_HOME="/gypsum/work1/zamani/hzeng/.cache/huggingface/"
export TRANSFORMERS_CACHE="/gypsum/work1/zamani/hzeng/.cache/huggingface/"
export HF_DATASETS_CACHE="/gypsum/work1/zamani/hzeng/.cache/huggingface/datasets"
mkdir -p "${HF_HOME}"

# =========================
# Model & Data paths
# =========================
CHECKPOINT_PATH="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="['${PROJECT_ROOT}/data/co_search/nq_40_multihop_60_51K/cot/co_search_rl_51k.train.parquet']"
VAL_DATA="['${PROJECT_ROOT}/data/co_search/nq_40_multihop_60_51K/cot/co_search_26k.sample_eval.parquet']"
PROJECT_NAME="co_search"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/co_search"

# Config files (now under CoSearch/config/)
AGENT_LOOP_CONFIG="${PROJECT_ROOT}/config/co_search_agent_loop_config.yaml"

# Retrieval service URL — set this to your running retriever's host:port
# See README.md -> Step 3 for how to start the retriever server
RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://localhost:8000/retrieve}"

# Dynamically generate tool config so RETRIEVAL_SERVICE_URL is injected at runtime.
# (Static YAML files cannot reference shell variables.)
TOOL_CONFIG="/tmp/co_search_tool_config_${SLURM_JOB_ID:-local}.yaml"
cat > "${TOOL_CONFIG}" <<EOF
tools:
  - class_name: verl.tools.co_search_tool.CoSearchTool
    config:
      type: native
      retrieval_service_url: "${RETRIEVAL_SERVICE_URL}"
      timeout: 30
      max_retries: 3
      retry_delay: 1.0
      retry_backoff: 2.0
      default_top_n: 50
      default_top_m: 5
      hit_cutoffs: [1, 3, 5]
      tool_score_metric: "hit"
      trivial_answers: ["yes", "no", "true", "false"]
      format_penalty: -0.2
EOF
echo "Generated tool config: ${TOOL_CONFIG} (url=${RETRIEVAL_SERVICE_URL})"

# Reranker model
RERANKER_CHECKPOINT_PATH="Qwen/Qwen2.5-7B-Instruct"

# =========================
# Hardware config (prefer Slurm runtime values when available)
# =========================
NNODES="${SLURM_NNODES:-2}"
N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
TP_SIZE=1  # Tensor parallelism for actor model

# =========================
# Training hyperparameters
# =========================
N_ROLLOUTS=8
TEMPERATURE=1.0
TOTAL_EPOCHS=1
TRAIN_BATCH_SIZE=512
ACTOR_LR=1e-6
ACTOR_BATCH_SIZE=128
ACTOR_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MCRI_BATCH_SIZE_PER_GPU=2
ACTOR_LR_WARMUP_STEPS_RATIO=0.04
KL_LOSS_COEF=0.001
SAVE_FREQ=10
TEST_FREQ=20
RERANKER_SAMPLING_VAL_START_STEP=10000

# =========================
# Reward function (adjusted path from /workspace/ to PROJECT_ROOT)
# =========================
REWARD_FN_PATH="${PROJECT_ROOT}/verl/verl/utils/reward_score/search_qa_f1_with_format_penalty.py"
TRAIN_REWARD_FN="search_qa_f1_penalty_compute_score"
VAL_REWARD_FN="search_qa_f1_penalty_compute_score"
FORMAT_PENALTY=-0.2

# =========================
# UID Grouping & Score Assignment functions
# =========================
UID_GROUP_FN_PATH="${PROJECT_ROOT}/verl/verl/experimental/agent_loop/uid_group_functions.py"
UID_GROUP_FN_NAME="group_by_muid_ans_in_doc_subq_rougeL1"
UID_GROUP_THRESHOLD=0.8

SCORE_ASSIGN_FN_PATH="${PROJECT_ROOT}/verl/verl/experimental/agent_loop/score_assign_functions.py"
SCORE_ASSIGN_FN_NAME="sum_tool_agent_score_with_cond_threshold"
AGENT_THRESHOLD=0.8
COND_THRESHOLD=0.8
FILTER_NO_ANSWER_IN_DOCS=false  # set to true to filter reranker outputs where answer_in_docs=False

NUM_EXAMINE=0
VAL_NUM_EXAMINE=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EXP_NAME="co_search_cot_grpo_N${N_ROLLOUTS}_T${TEMPERATURE}_lr${ACTOR_LR}_fp${FORMAT_PENALTY}_klcoef${KL_LOSS_COEF}_Ts${TRAIN_BATCH_SIZE}_RuidG_${UID_GROUP_FN_NAME}_Th${UID_GROUP_THRESHOLD}_ScoreFn_${SCORE_ASSIGN_FN_NAME}_ToolTh${COND_THRESHOLD}_FltnoAns_${FILTER_NO_ANSWER_IN_DOCS}_ds_nq_40_multihop_60_51K_${TIMESTAMP}"

# =========================
# Ray cluster setup
# Single-node: ray.init() in Python creates a local cluster — no manual start needed.
# Multi-node:  start Ray head + workers via srun, then export RAY_ADDRESS so
#              Python's ray.init() connects to THIS cluster, not a new one.
# =========================
if [ "$NNODES" -gt 1 ]; then
    RAY_PORT=6379

    MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address \
        | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1)

    echo "=== Ray multi-node setup: master=${MASTER_NODE} (${MASTER_ADDR}:${RAY_PORT}) ==="

    srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" \
        ray start --head \
        --node-ip-address="$MASTER_ADDR" \
        --port=$RAY_PORT \
        --num-cpus="${SLURM_CPUS_PER_TASK:-64}" \
        --num-gpus="$N_GPUS_PER_NODE" \
        --block &
    sleep 15

    WORKER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2 | paste -sd ',')
    echo "=== Starting Ray workers on: ${WORKER_NODES} ==="
    srun --nodes=$((NNODES - 1)) --ntasks=$((NNODES - 1)) \
        --nodelist="$WORKER_NODES" \
        ray start \
        --address="$MASTER_ADDR:$RAY_PORT" \
        --num-cpus="${SLURM_CPUS_PER_TASK:-64}" \
        --num-gpus="$N_GPUS_PER_NODE" \
        --block &
    sleep 15

    # Without RAY_ADDRESS, ray.init() in Python starts a NEW local cluster
    # instead of connecting to the one we just started
    export RAY_ADDRESS="$MASTER_ADDR:$RAY_PORT"
    echo "=== RAY_ADDRESS=${RAY_ADDRESS} ==="
else
    echo "=== Single-node: ray.init() will create a local cluster ==="
fi

echo "=== Starting training... ==="

# =========================
# Run training
# =========================
cd "${PROJECT_ROOT}"

python main_co_search_ppo.py \
        algorithm.use_kl_in_reward=False \
        algorithm.adv_estimator=grpo \
        data.train_files=${TRAIN_DATA} \
        data.val_files=${VAL_DATA} \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.max_prompt_length=20480 \
        data.max_response_length=4096 \
        data.truncation='error' \
        actor_rollout_ref.model.path=${CHECKPOINT_PATH} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
        actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
        actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.max_model_len=24576 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MCRI_BATCH_SIZE_PER_GPU \
        actor_rollout_ref.rollout.prompt_length=20480 \
        actor_rollout_ref.rollout.response_length=4096 \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
        actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
        actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
        actor_rollout_ref.rollout.multi_turn.format=search_r1 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG} \
        actor_rollout_ref.rollout.agent.num_workers=8 \
        actor_rollout_ref.rollout.agent.default_agent_loop=co_search_agent \
        actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MCRI_BATCH_SIZE_PER_GPU \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU} \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${ACTOR_LR_WARMUP_STEPS_RATIO} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        critic.enable=False \
        reward_model.enable=False \
        reward_model.reward_manager=multiturn \
        reward_model.use_reward_loop=True \
        custom_reward_function.path="${REWARD_FN_PATH}" \
        custom_reward_function.name="${TRAIN_REWARD_FN}" \
        +custom_reward_function.reward_kwargs.format_penalty=${FORMAT_PENALTY} \
        trainer.nnodes=${NNODES} \
        trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
        trainer.total_epochs=${TOTAL_EPOCHS} \
        trainer.experiment_name=${EXP_NAME} \
        trainer.max_actor_ckpt_to_keep=1 \
        +trainer.num_examine=${NUM_EXAMINE} \
        +trainer.val_num_examine=${VAL_NUM_EXAMINE} \
        trainer.val_before_train=False \
        trainer.logger=['console','wandb'] \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.default_local_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME}" \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.test_freq=${TEST_FREQ} \
        trainer.rollout_data_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME}"/rollout_data \
        trainer.validation_data_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME}"/validation_data \
        reranker_actor_rollout_ref.model.path=${RERANKER_CHECKPOINT_PATH} \
        reranker_actor_rollout_ref.model.use_remove_padding=True \
        reranker_actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
        reranker_actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
        reranker_actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
        reranker_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        reranker_actor_rollout_ref.rollout.max_model_len=24576 \
        reranker_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MCRI_BATCH_SIZE_PER_GPU \
        reranker_actor_rollout_ref.rollout.prompt_length=20480 \
        reranker_actor_rollout_ref.rollout.response_length=4096 \
        reranker_actor_rollout_ref.rollout.mode=async \
        reranker_actor_rollout_ref.rollout.name=vllm \
        reranker_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MCRI_BATCH_SIZE_PER_GPU \
        reranker_actor_rollout_ref.ref.fsdp_config.param_offload=False \
        reranker_actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
        reranker_actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_BATCH_SIZE} \
        reranker_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU} \
        reranker_actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
        reranker_actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${ACTOR_LR_WARMUP_STEPS_RATIO} \
        reranker_actor_rollout_ref.actor.use_kl_loss=True \
        reranker_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        reranker_actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        reranker_actor_rollout_ref.actor.fsdp_config.param_offload=False \
        reranker_actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        reranker_actor_rollout_ref.model.enable_gradient_checkpointing=True \
        reranker_actor_rollout_ref.trainable=True \
        reranker_uid_group_function.path="${UID_GROUP_FN_PATH}" \
        reranker_uid_group_function.name="${UID_GROUP_FN_NAME}" \
        +reranker_uid_group_function.uid_group_kwargs.threshold=${UID_GROUP_THRESHOLD} \
        reranker_score_assign_function.path="${SCORE_ASSIGN_FN_PATH}" \
        reranker_score_assign_function.name="${SCORE_ASSIGN_FN_NAME}" \
        +reranker_score_assign_function.score_assign_kwargs.agent_threshold=${AGENT_THRESHOLD} \
        +reranker_score_assign_function.score_assign_kwargs.cond_threshold=${COND_THRESHOLD} \
        trainer.reranker_sampling_val_start_step=${RERANKER_SAMPLING_VAL_START_STEP} \
        trainer.reranker_filter_no_answer_in_docs=${FILTER_NO_ANSWER_IN_DOCS}

# =========================
# Cleanup (only needed for multi-node where we manually started Ray)
# =========================
if [ "$NNODES" -gt 1 ]; then
    echo "=== Training complete. Stopping Ray cluster... ==="
    ray stop || true
fi
