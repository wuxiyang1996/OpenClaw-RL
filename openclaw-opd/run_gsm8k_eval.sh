#!/bin/bash
#
# GSM8K evaluation after OpenClaw-OPD LoRA training.
#
# Automatically picks up the base model and checkpoint paths from the
# training configuration.  Run from any directory:
#
#   bash openclaw-opd/run_gsm8k_eval.sh
#
# Override any default via environment variables:
#
#   NUM_PROBLEMS=200 TP=2 bash openclaw-opd/run_gsm8k_eval.sh
#
#   # Evaluate a specific checkpoint iteration instead of "latest":
#   ADAPTER_PATH=/path/to/ckpt/iter_0000005/model bash openclaw-opd/run_gsm8k_eval.sh
#
#   # Point at an already-running server (skip SGLang launch):
#   API_BASE=http://localhost:30000/v1 bash openclaw-opd/run_gsm8k_eval.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# ── Model paths (same defaults as run_qwen3_4b_openclaw_opd_topk_lora.sh) ──
HF_CKPT="${HF_CKPT:-${REPO_ROOT}/models/Qwen3-4B}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/ckpt/qwen3-4b-openclaw-opd-topk-lora}"

# Optional: path to a specific adapter directory (overrides auto-detect)
ADAPTER_PATH="${ADAPTER_PATH:-}"

# Optional: path to an already-merged model (skips LoRA merge entirely)
MERGED_MODEL="${MERGED_MODEL:-}"

# Where to save the merged model (empty = use a temp dir, cleaned up after)
MERGED_OUTPUT="${MERGED_OUTPUT:-}"

# ── Server settings ──
PORT="${PORT:-30050}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"

# If set, skip launching a local SGLang server and query this endpoint instead
API_BASE="${API_BASE:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-default}"

# ── Evaluation settings ──
DIFFICULTY="${DIFFICULTY:-easy}"               # "easy" = openai/gsm8k, "hard" = reasoning-machines/gsm-hard
NUM_PROBLEMS="${NUM_PROBLEMS:-1319}"          # full test set (both easy and hard have 1319)
CONCURRENCY="${CONCURRENCY:-32}"              # parallel requests
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
DATASET="${DATASET:-}"                         # empty = HuggingFace; or path to local JSON
TRAINING_MODE="${TRAINING_MODE:-}"            # set to "1" to send OPD training headers
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/gsm8k_${DIFFICULTY}_eval_results.json}"

export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# ── Validate base model ──
if [[ -z "${MERGED_MODEL}" ]]; then
  if [[ "${HF_CKPT}" == /* || "${HF_CKPT}" == ./* || "${HF_CKPT}" == ../* ]]; then
    if [[ ! -d "${HF_CKPT}" || ! -f "${HF_CKPT}/config.json" ]]; then
      echo "ERROR: Base model not found: ${HF_CKPT}"
      echo "  Download with: huggingface-cli download Qwen/Qwen3-4B --local-dir ${HF_CKPT}"
      exit 1
    fi
  fi
fi

# ── Build Python arguments ──
PYTHON_ARGS=()

if [[ -n "${MERGED_MODEL}" ]]; then
  PYTHON_ARGS+=(--model "${MERGED_MODEL}")
else
  PYTHON_ARGS+=(--base-model "${HF_CKPT}")
  if [[ -n "${ADAPTER_PATH}" ]]; then
    PYTHON_ARGS+=(--adapter "${ADAPTER_PATH}")
  else
    PYTHON_ARGS+=(--ckpt-dir "${CKPT_DIR}")
  fi
  if [[ -n "${MERGED_OUTPUT}" ]]; then
    PYTHON_ARGS+=(--merged-output "${MERGED_OUTPUT}")
  fi
fi

if [[ -n "${API_BASE}" ]]; then
  PYTHON_ARGS+=(--api-base "${API_BASE}")
else
  PYTHON_ARGS+=(--port "${PORT}" --tp "${TP}" --mem-fraction "${MEM_FRACTION}" --context-length "${CONTEXT_LENGTH}")
fi

PYTHON_ARGS+=(
  --served-model-name "${SERVED_MODEL_NAME}"
  --difficulty "${DIFFICULTY}"
  --num-problems "${NUM_PROBLEMS}"
  --concurrency "${CONCURRENCY}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
  --output "${OUTPUT}"
)

if [[ -n "${DATASET}" ]]; then
  PYTHON_ARGS+=(--dataset "${DATASET}")
fi

if [[ "${TRAINING_MODE}" == "1" ]]; then
  PYTHON_ARGS+=(--training-mode)
fi

echo "============================================================"
echo "  GSM8K Evaluation for OpenClaw-OPD LoRA"
echo "============================================================"
echo "  Difficulty:    ${DIFFICULTY}"
echo "  Base model:    ${MERGED_MODEL:-${HF_CKPT}}"
echo "  Checkpoint:    ${ADAPTER_PATH:-${CKPT_DIR} (latest)}"
echo "  Num problems:  ${NUM_PROBLEMS}"
echo "  Temperature:   ${TEMPERATURE}"
echo "  Training mode: ${TRAINING_MODE:-off}"
echo "  Output:        ${OUTPUT}"
echo "============================================================"
echo

exec python3 "${SCRIPT_DIR}/eval_gsm8k.py" "${PYTHON_ARGS[@]}"
