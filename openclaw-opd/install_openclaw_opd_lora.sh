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

# --- PyTorch with CUDA 12.8 ---
# Match your driver: change cu128 to cu124 or cu121 if needed.
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# --- Core training dependencies (slime/requirements.txt + extras) ---
pip install \
  accelerate==1.13.0 \
  blobfile \
  datasets \
  "httpx[http2]" \
  omegaconf \
  pillow \
  pylatexenc \
  pyyaml \
  "ray[default]==2.54.0" \
  tensorboard \
  transformers==4.57.1 \
  wandb

# LoRA support
pip install "peft>=0.18.0"

# Flash attention: slime uses ring_flash_attn
pip install ring_flash_attn 2>/dev/null || echo "WARN: ring_flash_attn failed (optional)"

# --- SGLang stack (rollout + inference engine) ---
# sglang-router: load-balancing router
pip install "sglang-router>=0.3.2"
# Full sglang package (not just sglang-router); repo-pinned ref to match OpenClaw-RL.
# --no-deps avoids overwriting torch.
pip install "git+https://github.com/sgl-project/sglang.git@dce8b0606c06d3a191a24c7b8cbe8e238ab316c9#egg=sglang&subdirectory=python" --no-deps

# flashinfer: required by sglang for attention backend (flashinfer is the default)
pip install "flashinfer-python>=0.6.3"

# xgrammar: required by sglang for constrained decoding / grammar backend
pip install "xgrammar>=0.1.27"

# outlines: alternative grammar backend used by sglang
pip install outlines

# --- OpenAI client (for sending requests to the OPD server) ---
pip install openai

# Optional for OPD API
pip install qwen_vl_utils 2>/dev/null || true

# Install slime in editable mode from repo
cd "${SLIME_ROOT}"
pip install -e . --no-deps
cd "${REPO_ROOT}"

# OpenClaw OPD API server deps
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
