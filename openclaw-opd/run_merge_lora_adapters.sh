#!/bin/bash
#
# Merge (weighted average) LoRA adapters from a dual-LoRA training checkpoint.
#
# Usage:
#   # Equal average from latest dual-LoRA checkpoint
#   bash openclaw-opd/run_merge_lora_adapters.sh
#
#   # Weighted average: 70% lora_a, 30% lora_b
#   WEIGHTS="0.7 0.3" bash openclaw-opd/run_merge_lora_adapters.sh
#
#   # Custom checkpoint and output paths
#   CKPT_DIR=/path/to/dual-lora-ckpt OUTPUT=/path/to/merged \
#     bash openclaw-opd/run_merge_lora_adapters.sh
#
#   # Merge two explicit adapter directories
#   ADAPTERS="/path/to/adapter_a /path/to/adapter_b" \
#     WEIGHTS="0.6 0.4" \
#     bash openclaw-opd/run_merge_lora_adapters.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/ckpt/qwen3-4b-openclaw-opd-topk-dual-lora}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/ckpt/qwen3-4b-openclaw-opd-merged-lora}"
ADAPTERS="${ADAPTERS:-}"
WEIGHTS="${WEIGHTS:-}"

PYTHON_ARGS=()

if [[ -n "${ADAPTERS}" ]]; then
  PYTHON_ARGS+=(--adapters ${ADAPTERS})
else
  PYTHON_ARGS+=(--ckpt-dir "${CKPT_DIR}")
fi

PYTHON_ARGS+=(--output "${OUTPUT}")

if [[ -n "${WEIGHTS}" ]]; then
  PYTHON_ARGS+=(--weights ${WEIGHTS})
fi

echo "============================================================"
echo "  Merge LoRA Adapters"
echo "============================================================"
echo "  Checkpoint: ${CKPT_DIR}"
echo "  Output:     ${OUTPUT}"
echo "  Weights:    ${WEIGHTS:-equal}"
echo "============================================================"
echo

exec python3 "${SCRIPT_DIR}/merge_lora_adapters.py" "${PYTHON_ARGS[@]}"
