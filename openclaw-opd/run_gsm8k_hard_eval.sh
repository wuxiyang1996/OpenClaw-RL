#!/bin/bash
#
# GSM8K HARD evaluation — reasoning-machines/gsm-hard (1319 problems).
#
# Same questions as standard GSM8K but with larger, less common numbers.
# Tests whether the model truly reasons rather than pattern-matches.
#
# Wrapper around run_gsm8k_eval.sh with DIFFICULTY=hard.
# All environment variable overrides from run_gsm8k_eval.sh are supported.
#
# Usage:
#   bash openclaw-opd/run_gsm8k_hard_eval.sh
#
#   # Against running dual-LoRA server (adapter B):
#   API_BASE=http://localhost:30001/v1 bash openclaw-opd/run_gsm8k_hard_eval.sh
#
#   # Evaluate a specific checkpoint:
#   ADAPTER_PATH=/path/to/ckpt/iter_0000003/model bash openclaw-opd/run_gsm8k_hard_eval.sh
#

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export DIFFICULTY="hard"
export OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/gsm8k_hard_eval_results.json}"

exec bash "${SCRIPT_DIR}/run_gsm8k_eval.sh"
