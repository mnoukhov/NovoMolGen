#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ai2/cuda12.8-dev-ubuntu22.04-notorch}"
WEKA_MOUNT="${WEKA_MOUNT:-oe-adapt-default:/weka/oe-adapt-default}"
DEFAULT_PYTHON="${DEFAULT_PYTHON:-3.10}"
PRIORITY="${PRIORITY:-high}"
CLUSTER="${CLUSTER:-ai2/saturn}"
WORKSPACE="${WORKSPACE:-ai2/oe-adapt-code}"
GPUS="${GPUS:-4}"

WANDB_ENTITY="${WANDB_ENTITY:-diegos-mila}"
WANDB_PROJECT="${WANDB_PROJECT:-chem-llm-pipeline}"
WANDB_MODE="${WANDB_MODE:-online}"
CONFIG_NAME="${CONFIG_NAME:-finetune_PMO_ZINC_1B_bpe_smiles_llama-32M}"
GANTRY_GROUP="${GANTRY_GROUP:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_name> [extra finetune args...]" >&2
  exit 1
fi

task="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

group_args=()
if [[ -n "$GANTRY_GROUP" ]]; then
  group_args+=(--group "$GANTRY_GROUP")
fi

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo no-git-branch)"
branch="${branch//\//-}"
short_commit="$(git rev-parse --short HEAD 2>/dev/null || echo no-git-commit)"
ts="$(date -u +%Y%m%d-%H%M%S)"
slug="$(printf '%s' "$task" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9_-')"
name="novomol-${slug}-${ts}"

echo "Submitting task: $task"
echo "Gantry name: $name"

gantry run \
    --beaker-image "$IMAGE" \
    --weka="$WEKA_MOUNT" \
    --uv-all-extras \
    --show-logs \
    --default-python-version "$DEFAULT_PYTHON" \
    --env WANDB_ENTITY="$WANDB_ENTITY" \
    --env WANDB_PROJECT="$WANDB_PROJECT" \
    --env WANDB_MODE="$WANDB_MODE" \
    --env WANDB_TAGS="${branch},${short_commit},${slug}" \
    --secret-env HF_TOKEN=michaeln_HF_TOKEN \
    --secret-env WANDB_API_KEY=michaeln_WANDB_API_KEY \
    --priority "$PRIORITY" \
    --cluster "$CLUSTER" \
    --workspace "$WORKSPACE" \
    --gpus "$GPUS" \
    --name "$name" \
    "${group_args[@]}" \
    -- \
    uv run src/main.py finetune \
      --config_name="$CONFIG_NAME" \
      --finetune.task_name="$task" \
      "$@"
