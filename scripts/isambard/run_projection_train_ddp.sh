#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7
module load brics/nccl
module load cudatoolkit/24.11_12.6

export GRAPHMED_HOME="${GRAPHMED_HOME:-$HOME/GraphMed-LT}"
cd "$GRAPHMED_HOME"
source .venv/bin/activate

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$GRAPHMED_HOME}/GraphMed-LT/.hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TRAIN_FILE="${TRAIN_FILE:-data/all_train_convo.jsonl}"
EXPERT_MODEL="${EXPERT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TRIPLET_MODEL="${TRIPLET_MODEL:-llama-3.3-70b-instruct-awq}"
TRIPLET_CORPUS="${TRIPLET_CORPUS:-$HOME/SEE-Bio/data/primekg_triplets.jsonl}"
SAVE_CKPT="${SAVE_CKPT:-save_model/graphmed_lt.ckpt}"
SAVE_DOCTOR_DIR="${SAVE_DOCTOR_DIR:-save_model/doctor_agent}"

PREFIX_LEN="${PREFIX_LEN:-20}"
REFINEMENT_STEPS="${REFINEMENT_STEPS:-5}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GNN_IN_DIM="${GNN_IN_DIM:-256}"
GNN_HIDDEN_DIM="${GNN_HIDDEN_DIM:-256}"
GNN_LAYERS="${GNN_LAYERS:-2}"
GAT_HEADS="${GAT_HEADS:-4}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-3}"
MAX_CORPUS_TRIPLETS="${MAX_CORPUS_TRIPLETS:-}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
NNODES="${SLURM_NNODES:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
fi
export MASTER_ADDR MASTER_PORT GPUS_PER_NODE

COMMON_ARGS=(
  projection_train.py
  --train_file "$TRAIN_FILE"
  --expert_model "$EXPERT_MODEL"
  --triplet_model "$TRIPLET_MODEL"
  --triplet_corpus "$TRIPLET_CORPUS"
  --retrieval_top_k "$RETRIEVAL_TOP_K"
  --prefix_len "$PREFIX_LEN"
  --refinement_steps "$REFINEMENT_STEPS"
  --gnn_model gat
  --gnn_in_dim "$GNN_IN_DIM"
  --gnn_hidden_dim "$GNN_HIDDEN_DIM"
  --gnn_layers "$GNN_LAYERS"
  --gat_heads "$GAT_HEADS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --save_ckpt "$SAVE_CKPT"
  --save_doctor_dir "$SAVE_DOCTOR_DIR"
  --seed "$SEED"
  --gradient_checkpointing
)

if [ -n "$MAX_CORPUS_TRIPLETS" ]; then
  COMMON_ARGS+=(--max_corpus_triplets "$MAX_CORPUS_TRIPLETS")
fi

if [ "${USE_WANDB:-0}" = "1" ]; then
  COMMON_ARGS+=(--use_wandb)
fi

if [ "$NNODES" -gt 1 ]; then
  export GRAPHMED_TRAIN_ARGS="$(printf '%q ' "${COMMON_ARGS[@]}") $EXTRA_ARGS"
  srun --ntasks="$NNODES" --ntasks-per-node=1 bash -lc '
    cd "$GRAPHMED_HOME"
    source .venv/bin/activate
    python -m torch.distributed.run \
      --nnodes "$SLURM_NNODES" \
      --nproc_per_node "$GPUS_PER_NODE" \
      --node_rank "$SLURM_PROCID" \
      --master_addr "$MASTER_ADDR" \
      --master_port "$MASTER_PORT" \
      $GRAPHMED_TRAIN_ARGS
  '
else
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$GPUS_PER_NODE" \
    "${COMMON_ARGS[@]}" \
    $EXTRA_ARGS
fi
