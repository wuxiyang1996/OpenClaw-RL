#!/bin/bash
#
# GSM8K EASY evaluation — standard openai/gsm8k test set (1319 problems).
#
# Wrapper around run_gsm8k_eval.sh with DIFFICULTY=easy.
# All environment variable overrides from run_gsm8k_eval.sh are supported.
#
# Usage:
#   bash openclaw-opd/run_gsm8k_easy_eval.sh
#
#   # Against running dual-LoRA server (adapter A):
#   API_BASE=http://localhost:30000/v1 bash openclaw-opd/run_gsm8k_easy_eval.sh
#
#   # Feed training data into a running dual-LoRA server (adapter A):
#   API_BASE=http://localhost:30000/v1 TRAINING_MODE=1 bash openclaw-opd/run_gsm8k_easy_eval.sh
#
#   # Evaluate a specific checkpoint:
#   ADAPTER_PATH=/path/to/ckpt/iter_0000003/model bash openclaw-opd/run_gsm8k_easy_eval.sh
#

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export DIFFICULTY="easy"
export OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/gsm8k_easy_eval_results.json}"

exec bash "${SCRIPT_DIR}/run_gsm8k_eval.sh"
