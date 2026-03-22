#!/bin/bash
#
# Federated dual-LoRA training on 4 GPUs.
#
# Two OpenClawOPDAPIServer instances run on PORT_A and PORT_B, each tagged
# with its own adapter name. SGLang serves both adapters via native multi-LoRA.
# A single FSDP actor trains both adapters each step (sequential fwd/bwd,
# single optimizer.step).
#

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

NUM_GPUS=${NUM_GPUS:-4}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
PRM_GPUS=${PRM_GPUS:-1}

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

HF_CKPT=${HF_CKPT:-${REPO_ROOT}/models/Qwen3-4B}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-${REPO_ROOT}/ckpt/qwen3-4b-openclaw-opd-topk-dual-lora}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}

# Resume from a previous checkpoint (single-LoRA or dual-LoRA).
# A single-LoRA checkpoint is automatically duplicated into both adapters.
# Set to a dual-LoRA ckpt dir or a single-LoRA ckpt dir, or leave empty.
LOAD_CKPT="${LOAD_CKPT:-}"

if [[ "${HF_CKPT}" == /* || "${HF_CKPT}" == ./* || "${HF_CKPT}" == ../* ]]; then
  if [[ ! -d "${HF_CKPT}" || ! -f "${HF_CKPT}/config.json" ]]; then
    echo "ERROR: HF checkpoint path not found or invalid: ${HF_CKPT}"
    echo "  Download with: huggingface-cli download Qwen/Qwen3-4B --local-dir ${HF_CKPT}"
    exit 1
  fi
fi

# --- Dual LoRA configuration ---
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
ADAPTER_NAME_A=${ADAPTER_NAME_A:-"lora_a"}
ADAPTER_NAME_B=${ADAPTER_NAME_B:-"lora_b"}
PORT_A=${PORT_A:-30000}
PORT_B=${PORT_B:-30001}

export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
# PORT_A / PORT_B are passed via runtime env to the rollout worker
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen3_4b_dual_lora_record.jsonl"
export TP="${TP:-1}"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.85"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
export PRM_M="${PRM_M:-3}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-3}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 1
)

if [[ -n "${LOAD_CKPT}" ]]; then
  CKPT_ARGS+=(--load "${LOAD_CKPT}" --no-load-optim --no-load-rng --finetune)
  echo "INFO: Resuming from checkpoint: ${LOAD_CKPT}"
  echo "  Single-LoRA checkpoints will be duplicated into both adapters."
fi

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_opd_rollout.generate_rollout_openclaw_opd

   --num-rollout 100000000
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 0.6
   --reward-key score

   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
   --gradient-checkpointing
   --attn-implementation "${ATTN_IMPL}"
)

OPD_ARGS=(
   --loss-type custom_loss
   --custom-loss-function-path topk_distillation_loss.topk_distillation_loss_function
   --distill-topk 50
   --disable-compute-advantages-and-returns
   --disable-rewards-normalization
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

LORA_ARGS=(
   --use-lora
   --lora-rank "${LORA_RANK}"
   --lora-alpha "${LORA_ALPHA}"
   --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
   --num-lora-adapters 2
   --lora-adapter-names "${ADAPTER_NAME_A},${ADAPTER_NAME_B}"
)

EVAL_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${TP}"
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.85
   --sglang-context-length 32768
   --sglang-reasoning-parser qwen3
)

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine "${PRM_TP:-${TP}}"
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path openclaw_opd_api_server.generate
   --custom-rm-path openclaw_opd_api_server.reward_func
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

PYTHON_EXE="${PYTHON_EXE:-$(which python3)}"
if [[ -z "${PYTHON_EXE}" ]]; then
  echo "ERROR: python3 not found. Activate the openclaw-opd-lora conda env and re-run."
  exit 1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
    \"PORT_A\": \"${PORT_A}\",
    \"PORT_B\": \"${PORT_B}\",
    \"ADAPTER_NAME_A\": \"${ADAPTER_NAME_A}\",
    \"ADAPTER_NAME_B\": \"${ADAPTER_NAME_B}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir "${SLIME_ROOT}" \
   -- "${PYTHON_EXE}" "${SLIME_ROOT}/train_async_federated.py" \
   --train-backend fsdp \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${OPD_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]} \
   ${LORA_ARGS[@]}
