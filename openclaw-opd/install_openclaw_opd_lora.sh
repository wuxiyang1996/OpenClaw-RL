#!/bin/bash
# Install environment for OpenClaw OPD with Qwen3-4B + LoRA.
# Run from repo root: bash openclaw-opd/install_openclaw_opd_lora.sh
# Then: conda activate openclaw-opd-lora
#       cd slime && bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh  (after setting HF_CKPT)

set -e

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPO_ROOT}/slime"
OPENCLAW_OPD_ROOT="${REPO_ROOT}/openclaw-opd"
ENV_NAME="${OPENCLAW_OPD_ENV_NAME:-openclaw-opd-lora}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "SLIME_ROOT=${SLIME_ROOT}"
echo "Creating conda env: ${ENV_NAME}"

# Create conda env with Python 3.10 (no CUDA from conda to avoid version clashes with PyTorch)
conda create -n "${ENV_NAME}" python=3.10 -y -c conda-forge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.4 (match your driver; cu121/cu124/cu128)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install slime dependencies (from slime/requirements.txt)
pip install \
  accelerate \
  blobfile \
  datasets \
  "httpx[http2]" \
  omegaconf \
  pillow \
  pylatexenc \
  pyyaml \
  "ray[default]" \
  tensorboard \
  transformers \
  wandb

# LoRA support
pip install "peft>=0.12.0"

# Flash attention: slime uses ring_flash_attn
pip install ring_flash_attn 2>/dev/null || echo "WARN: ring_flash_attn failed (optional)"

# SGLang for rollout (required by openclaw-opd)
pip install "sglang-router>=0.2.3"
pip install sglang 2>/dev/null || pip install "sglang[all]" 2>/dev/null || echo "WARN: Install sglang from source if needed: cd sglang && pip install -e 'python[all]'"

# Optional for OPD API
pip install qwen_vl_utils 2>/dev/null || true

# Install slime in editable mode from repo
cd "${SLIME_ROOT}"
pip install -e . --no-deps
cd "${REPO_ROOT}"

# OpenClaw OPD API server deps (already in slime/requirements or common)
pip install httpx fastapi uvicorn

echo ""
echo "Done. Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Before running the LoRA script:"
echo "  1. Set HF_CKPT to your Qwen3-4B path, e.g.:"
echo "     export HF_CKPT=${REPO_ROOT}/models/Qwen3-4B"
echo "  2. Download model if needed:"
echo "     huggingface-cli download Qwen/Qwen3-4B --local-dir ${REPO_ROOT}/models/Qwen3-4B"
echo "  3. From slime dir: bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh"
echo ""
echo "Note: run_qwen3_4b_openclaw_opd_topk_lora.sh uses custom top-K loss which imports megatron.core."
echo "If you see ModuleNotFoundError for megatron, either add Megatron-LM to PYTHONPATH or use"
echo "token-level OPD: run_qwen3_4b_openclaw_opd.sh (with Megatron backend)."
