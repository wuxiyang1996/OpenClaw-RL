# Use Cases

Copy-paste-ready command sequences for the most common workflows.
All commands assume `conda activate openclaw-opd-lora` and run from the
**repo root** (`/workspace/OpenClaw-RL`).

---

## Use Case 1: Dual-LoRA Training

Train two independent LoRA adapters on the same base model with strict
port-to-adapter separation. Both train concurrently on 4 GPUs.

### 1a. From scratch

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

Two API servers come up:

| Port | Adapter | Serves |
|------|---------|--------|
| 30000 | `lora_a` | Requests to `:30000` always use `lora_a` |
| 30001 | `lora_b` | Requests to `:30001` always use `lora_b` |

Send multi-turn conversations to either port to produce OPD training samples.
Each adapter only trains on data from its own port.

### 1b. From an existing single-LoRA checkpoint

The single-LoRA weights are automatically **duplicated** into both `lora_a`
and `lora_b` as the starting point — both adapters begin identical, then
diverge as they receive different training data.

```bash
cd slime
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-lora \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

### 1c. Override adapter names and ports

```bash
cd slime
PORT_A=40000 PORT_B=40001 \
ADAPTER_NAME_A=client_x ADAPTER_NAME_B=client_y \
LORA_RANK=32 LORA_ALPHA=64 \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

### What it produces

```text
ckpt/qwen3-4b-openclaw-opd-topk-dual-lora/
├── latest_checkpointed_iteration.txt
└── iter_0000001/
    ├── model/
    │   ├── lora_a/adapter_weights.pt    ← lora_a weights
    │   └── lora_b/adapter_weights.pt    ← lora_b weights
    └── meta.json
```

---

## Use Case 2: GSM8K Evaluation (Easy + Hard, in Parallel)

Evaluate a checkpoint on both the **standard GSM8K** (1319 problems) and
**GSM-Hard** (same problems with larger numbers) simultaneously.

### 2a. Against a running dual-LoRA training server (no extra GPU)

While dual-LoRA training is running, the SGLang engine already serves both
adapters. Evaluate each adapter on a different difficulty:

```bash
# lora_a on easy GSM8K, lora_b on hard GSM8K — runs in parallel
API_BASE=http://localhost:30000 bash openclaw-opd/run_gsm8k_easy_eval.sh &
API_BASE=http://localhost:30001 bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait

echo "Easy results:  openclaw-opd/results/gsm8k_easy_eval_results.json"
echo "Hard results:  openclaw-opd/results/gsm8k_hard_eval_results.json"
```

### 2b. Against a running single-LoRA training server

```bash
# Same adapter, both difficulties
API_BASE=http://localhost:30000 bash openclaw-opd/run_gsm8k_easy_eval.sh &
API_BASE=http://localhost:30000 bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### 2c. Launch dedicated eval servers from a checkpoint (offline eval)

Each eval script launches its own SGLang server, merges the LoRA into the base
model, and evaluates. Use different ports to avoid collisions:

```bash
# From the latest single-LoRA checkpoint
PORT=30050 bash openclaw-opd/run_gsm8k_easy_eval.sh &
PORT=30051 bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### 2d. Evaluate each dual-LoRA adapter from a saved checkpoint

```bash
CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-dual-lora

# Find latest iteration
ITER=$(cat "${CKPT}/latest_checkpointed_iteration.txt")
MODEL_DIR="${CKPT}/iter_$(printf '%07d' ${ITER})/model"

# Evaluate lora_a on easy, lora_b on hard
ADAPTER_PATH="${MODEL_DIR}/lora_a" PORT=30050 \
  bash openclaw-opd/run_gsm8k_easy_eval.sh &
ADAPTER_PATH="${MODEL_DIR}/lora_b" PORT=30051 \
  bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### 2e. Custom output paths

```bash
API_BASE=http://localhost:30000/v1 \
OUTPUT=openclaw-opd/results/experiment1_easy.json \
  bash openclaw-opd/run_gsm8k_easy_eval.sh &

API_BASE=http://localhost:30001/v1 \
OUTPUT=openclaw-opd/results/experiment1_hard.json \
  bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### Quick reference: eval env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE` | (empty) | Point at existing server; skips launching SGLang |
| `PORT` | 30050 | SGLang port when launching a new server |
| `ADAPTER_PATH` | (empty) | Specific adapter dir to evaluate |
| `CKPT_DIR` | `ckpt/qwen3-4b-openclaw-opd-topk-lora` | Auto-detect latest checkpoint |
| `NUM_PROBLEMS` | 1319 | Number of problems to evaluate |
| `OUTPUT` | `results/gsm8k_{easy,hard}_eval_results.json` | Results file |

---

## Use Case 3: Merge Adapters and Start New Training

After dual-LoRA training, average the two adapters into one, then use
the merged result as the starting point for the next round.

### Step 1: Merge the two adapters

```bash
# Equal average (50/50)
bash openclaw-opd/run_merge_lora_adapters.sh
```

Or with custom weights (e.g., 70% lora_a + 30% lora_b):

```bash
WEIGHTS="0.7 0.3" bash openclaw-opd/run_merge_lora_adapters.sh
```

This reads the latest dual-LoRA checkpoint from
`ckpt/qwen3-4b-openclaw-opd-topk-dual-lora/` and writes the merged adapter to
`ckpt/qwen3-4b-openclaw-opd-merged-lora/`.

### Step 2: Evaluate the merged adapter (optional)

```bash
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
PORT=30050 bash openclaw-opd/run_gsm8k_easy_eval.sh &

ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
PORT=30051 bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### Step 3: Start a new round of dual-LoRA training from the merged adapter

The merged adapter is duplicated into both `lora_a` and `lora_b` — they start
identical, then diverge again as they receive different data:

```bash
cd slime
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
SAVE_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-dual-lora-round2 \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

### Full iterative loop

```bash
cd slime

# ── Round 1 ──
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh

# ── Merge round 1 ──
cd ..
WEIGHTS="0.5 0.5" bash openclaw-opd/run_merge_lora_adapters.sh

# ── Evaluate merged ──
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
  bash openclaw-opd/run_gsm8k_easy_eval.sh

# ── Round 2 from merged ──
cd slime
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
SAVE_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-dual-lora-round2 \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh

# ── Merge round 2 ──
cd ..
CKPT_DIR=/workspace/OpenClaw-RL/ckpt/qwen3-4b-dual-lora-round2 \
OUTPUT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-merged-round2 \
WEIGHTS="0.5 0.5" bash openclaw-opd/run_merge_lora_adapters.sh

# ── Evaluate round 2 ──
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-merged-round2 \
  bash openclaw-opd/run_gsm8k_easy_eval.sh
```

### Merge quick reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CKPT_DIR` | `ckpt/qwen3-4b-openclaw-opd-topk-dual-lora` | Source dual-LoRA checkpoint |
| `OUTPUT` | `ckpt/qwen3-4b-openclaw-opd-merged-lora` | Where to save merged adapter |
| `WEIGHTS` | (equal) | Space-separated weights, e.g. `"0.7 0.3"` |
| `ADAPTERS` | (empty) | Explicit adapter dirs (overrides `CKPT_DIR`) |

---

## Summary: Which Script Does What

| Script | What it does |
|--------|-------------|
| `run_qwen3_4b_openclaw_opd_topk_dual_lora.sh` | Train two LoRA adapters on 4 GPUs |
| `run_gsm8k_easy_eval.sh` | Evaluate on standard GSM8K (1319 problems) |
| `run_gsm8k_hard_eval.sh` | Evaluate on GSM-Hard (larger numbers) |
| `run_merge_lora_adapters.sh` | Weighted average of LoRA adapters |

For full configuration reference, see [`usage-guide.md`](usage-guide.md).
