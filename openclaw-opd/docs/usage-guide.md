# Usage Guide: OpenClaw-OPD Training & Evaluation

End-to-end guide for running single-LoRA training, dual-LoRA (federated)
training, adapter merging, and GSM8K evaluation.

---

## Prerequisites

```bash
# 1. Create and activate the conda environment
bash openclaw-opd/install_openclaw_opd_lora.sh
conda activate openclaw-opd-lora

# 2. Download the base model
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir /workspace/OpenClaw-RL/models/Qwen3-4B
```

All commands below assume the conda env is activated and run from the repo
root (or `slime/` where noted).

---

## 1. Single-LoRA Training (Option C)

Train one LoRA adapter with FSDP on 4 GPUs.

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_CKPT` | `models/Qwen3-4B` | Base model path or HF repo id |
| `NUM_GPUS` | 4 | Total GPUs |
| `ACTOR_GPUS` | 2 | GPUs for FSDP actor |
| `ROLLOUT_GPUS` | 1 | GPUs for SGLang inference |
| `PRM_GPUS` | 1 | GPUs for PRM judge |
| `ATTN_IMPL` | `sdpa` | Attention backend (`sdpa`, `flash_attention_2`, `eager`) |

The API server listens on port **30000**. Send multi-turn conversations to
produce OPD training samples (see [Sending Requests](#6-sending-requests)).

Checkpoints are saved to `ckpt/qwen3-4b-openclaw-opd-topk-lora/`.

---

## 2. Dual-LoRA Training (Option D — Federated)

Train two distinct LoRA adapters concurrently on 4 GPUs with strict port-to-adapter separation.

### From scratch

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

Two API servers start:
- Port **30000** → `lora_a`
- Port **30001** → `lora_b`

### From a previous single-LoRA checkpoint

The existing LoRA weights are duplicated into both `lora_a` and `lora_b` as the starting point:

```bash
cd slime
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-lora \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

### From a merged adapter

After merging two adapters (step 4), use the merged result as a new starting point:

```bash
cd slime
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

### Resume from a previous dual-LoRA checkpoint

Each adapter loads its own weights from the sub-directories:

```bash
cd slime
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-dual-lora \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT_A` | 30000 | API port for adapter A |
| `PORT_B` | 30001 | API port for adapter B |
| `ADAPTER_NAME_A` | `lora_a` | PEFT name for adapter A |
| `ADAPTER_NAME_B` | `lora_b` | PEFT name for adapter B |
| `LORA_RANK` | 16 | LoRA rank (shared) |
| `LORA_ALPHA` | 32 | LoRA alpha (shared) |
| `LOAD_CKPT` | (empty) | Previous checkpoint to resume from |

---

## 3. GSM8K Evaluation

Evaluate a trained checkpoint on GSM8K (easy) or GSM-Hard (hard, larger numbers).

### Launch a new server and evaluate

```bash
# Easy (original GSM8K — 1319 problems)
bash openclaw-opd/run_gsm8k_easy_eval.sh

# Hard (GSM-Hard — same problems, bigger numbers)
bash openclaw-opd/run_gsm8k_hard_eval.sh
```

### Run both in parallel

```bash
PORT=30050 bash openclaw-opd/run_gsm8k_easy_eval.sh &
PORT=30051 bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### Evaluate against a running training server (no extra GPU)

During training, the SGLang engine is already up. Point eval at it directly:

```bash
# Single-LoRA server on port 30000
API_BASE=http://localhost:30000/v1 bash openclaw-opd/run_gsm8k_easy_eval.sh

# Dual-LoRA: evaluate each adapter on a different difficulty
API_BASE=http://localhost:30000/v1 bash openclaw-opd/run_gsm8k_easy_eval.sh &
API_BASE=http://localhost:30001/v1 bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### Evaluate a specific checkpoint

```bash
# Single-LoRA checkpoint
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-lora/iter_0000005/model \
  bash openclaw-opd/run_gsm8k_easy_eval.sh

# Dual-LoRA: evaluate each adapter separately
ADAPTER_PATH=.../iter_0000003/model/lora_a bash openclaw-opd/run_gsm8k_easy_eval.sh &
ADAPTER_PATH=.../iter_0000003/model/lora_b bash openclaw-opd/run_gsm8k_hard_eval.sh &
wait
```

### Evaluate a merged adapter

```bash
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
  bash openclaw-opd/run_gsm8k_easy_eval.sh
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DIFFICULTY` | `easy` | `easy` = openai/gsm8k, `hard` = reasoning-machines/gsm-hard |
| `NUM_PROBLEMS` | 1319 | Number of problems |
| `CONCURRENCY` | 32 | Parallel requests |
| `TEMPERATURE` | 0.0 | Greedy decoding |
| `PORT` | 30050 | SGLang server port (when launching a new server) |
| `API_BASE` | (empty) | Use an existing server (skips launching SGLang) |
| `ADAPTER_PATH` | (empty) | Specific adapter dir (overrides auto-detect) |
| `CKPT_DIR` | `ckpt/qwen3-4b-openclaw-opd-topk-lora` | Checkpoint dir (auto-detects latest) |

Results are saved to:
- `openclaw-opd/results/gsm8k_easy_eval_results.json`
- `openclaw-opd/results/gsm8k_hard_eval_results.json`

---

## 4. Merging LoRA Adapters

Average two LoRA adapters from a dual-LoRA checkpoint into a single adapter.

### Equal average (default)

```bash
bash openclaw-opd/run_merge_lora_adapters.sh
```

### Weighted average

```bash
WEIGHTS="0.7 0.3" bash openclaw-opd/run_merge_lora_adapters.sh
```

### Custom paths

```bash
CKPT_DIR=/path/to/dual-lora-ckpt \
OUTPUT=/path/to/output \
  bash openclaw-opd/run_merge_lora_adapters.sh
```

### Merge two explicit adapter directories

```bash
ADAPTERS="/path/to/adapter_a /path/to/adapter_b" \
WEIGHTS="0.6 0.4" \
OUTPUT=/path/to/merged \
  bash openclaw-opd/run_merge_lora_adapters.sh
```

### Python directly

```bash
python openclaw-opd/merge_lora_adapters.py \
  --ckpt-dir /workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-dual-lora \
  --weights 0.7 0.3 \
  --output /workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora
```

| Variable | Default | Description |
|----------|---------|-------------|
| `CKPT_DIR` | `ckpt/qwen3-4b-openclaw-opd-topk-dual-lora` | Dual-LoRA checkpoint dir |
| `OUTPUT` | `ckpt/qwen3-4b-openclaw-opd-merged-lora` | Output directory |
| `WEIGHTS` | (equal) | Space-separated weights, e.g. `"0.7 0.3"` |
| `ADAPTERS` | (empty) | Explicit adapter paths (overrides `CKPT_DIR`) |

---

## 5. End-to-End Workflows

### Workflow A: Single-LoRA → Eval

```bash
cd slime

# Train
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh

# Evaluate (uses latest checkpoint)
bash ../openclaw-opd/run_gsm8k_easy_eval.sh
```

### Workflow B: Single-LoRA → Dual-LoRA → Merge → Eval

```bash
cd slime

# Step 1: Train single LoRA
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh

# Step 2: Start dual-LoRA from that checkpoint
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-lora \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh

# Step 3: Merge the two adapters (weighted)
cd ..
WEIGHTS="0.7 0.3" bash openclaw-opd/run_merge_lora_adapters.sh

# Step 4: Evaluate the merged adapter
ADAPTER_PATH=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
  bash openclaw-opd/run_gsm8k_easy_eval.sh
```

### Workflow C: Iterative Dual-LoRA (multiple rounds)

```bash
cd slime

# Round 1: train from scratch
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh

# Merge round 1
cd .. && WEIGHTS="0.5 0.5" bash openclaw-opd/run_merge_lora_adapters.sh && cd slime

# Round 2: resume from merged adapter
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
SAVE_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-dual-lora-round2 \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh

# Evaluate round 2
cd ..
CKPT_DIR=/workspace/OpenClaw-RL/ckpt/qwen3-4b-dual-lora-round2 \
  bash openclaw-opd/run_gsm8k_easy_eval.sh
```

### Workflow D: Compare Easy vs Hard during training

While training is running, evaluate both adapters live:

```bash
# In a separate terminal, while dual-LoRA training is running:
API_BASE=http://localhost:30000/v1 \
OUTPUT=openclaw-opd/results/lora_a_easy.json \
  bash openclaw-opd/run_gsm8k_easy_eval.sh &

API_BASE=http://localhost:30001/v1 \
OUTPUT=openclaw-opd/results/lora_b_hard.json \
  bash openclaw-opd/run_gsm8k_hard_eval.sh &

wait
echo "Results: lora_a_easy.json  lora_b_hard.json"
```

---

## 6. Sending Requests

The OPD server exposes an OpenAI-compatible chat completions API. Training
samples are created from multi-turn conversations — each `main` turn is
evaluated by the PRM judge when the next user message arrives.

### curl

```bash
# Health check
curl http://0.0.0.0:30000/healthz

# Turn 1 (main turn — will be evaluated when turn 2 arrives)
curl -X POST http://0.0.0.0:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: sess-001" \
  -H "X-Turn-Type: main" \
  -d '{
    "messages": [{"role": "user", "content": "Solve 2+2"}],
    "model": "qwen3-4b",
    "max_tokens": 2048
  }'

# Turn 2 (triggers OPD sample creation for turn 1)
curl -X POST http://0.0.0.0:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: sess-001" \
  -H "X-Turn-Type: main" \
  -d '{
    "messages": [
      {"role": "user", "content": "Solve 2+2"},
      {"role": "assistant", "content": "The answer is 4."},
      {"role": "user", "content": "Now solve 3+3"}
    ],
    "model": "qwen3-4b",
    "max_tokens": 2048
  }'

# End session
curl -X POST http://0.0.0.0:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: sess-001" \
  -H "X-Session-Done: true" \
  -d '{
    "messages": [
      {"role": "user", "content": "Solve 2+2"},
      {"role": "assistant", "content": "The answer is 4."},
      {"role": "user", "content": "Now solve 3+3"},
      {"role": "assistant", "content": "The answer is 6."}
    ],
    "session_done": true
  }'
```

### Python

```python
from openai import OpenAI

client = OpenAI(api_key="dummy_key", base_url="http://0.0.0.0:30000/v1")

resp1 = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Solve 2+2"}],
    max_tokens=2048,
    extra_headers={"X-Session-ID": "sess-001", "X-Turn-Type": "main"},
)

resp2 = client.chat.completions.create(
    model="qwen3-4b",
    messages=[
        {"role": "user", "content": "Solve 2+2"},
        {"role": "assistant", "content": resp1.choices[0].message.content},
        {"role": "user", "content": "Now solve 3+3"},
    ],
    max_tokens=2048,
    extra_headers={"X-Session-ID": "sess-001", "X-Turn-Type": "main"},
)
```

### Dual-LoRA: send to different ports

```python
from openai import OpenAI

client_a = OpenAI(api_key="dummy", base_url="http://0.0.0.0:30000/v1")
client_b = OpenAI(api_key="dummy", base_url="http://0.0.0.0:30001/v1")

# Client A → trains lora_a
resp_a = client_a.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Help from client A"}],
    extra_headers={"X-Session-ID": "sess-a-001", "X-Turn-Type": "main"},
)

# Client B → trains lora_b
resp_b = client_b.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Help from client B"}],
    extra_headers={"X-Session-ID": "sess-b-001", "X-Turn-Type": "main"},
)
```

### Headers

| Header | Values | Purpose |
|--------|--------|---------|
| `X-Session-ID` | any string | Groups turns into a session |
| `X-Turn-Type` | `main` / `side` | Only `main` turns produce OPD samples |
| `X-Session-Done` | `1` / `true` | End session, trigger cleanup |

### Notes

- Multi-turn conversations are required — OPD samples are created when the
  *next* message arrives, allowing PRM to judge the previous turn.
- The server returns **503** briefly during weight updates — retry after a
  short delay.
- Training begins once `rollout-batch-size` samples (default 4) accumulate.

---

## 7. Checkpoint Layout

### Single-LoRA checkpoint

```text
ckpt/qwen3-4b-openclaw-opd-topk-lora/
├── latest_checkpointed_iteration.txt    # e.g. "3"
├── iter_0000001/
│   ├── model/
│   │   ├── adapter_weights.pt           # LoRA weights
│   │   └── adapter_config.json          # LoRA config
│   ├── optimizer/                       # AdamW state
│   ├── lr_scheduler/                    # LR scheduler state
│   ├── rng.pt                           # RNG state
│   └── meta.json                        # Metadata
├── iter_0000002/
│   └── ...
└── iter_0000003/
    └── ...
```

### Dual-LoRA checkpoint

```text
ckpt/qwen3-4b-openclaw-opd-topk-dual-lora/
├── latest_checkpointed_iteration.txt
└── iter_0000001/
    ├── model/
    │   ├── lora_a/
    │   │   ├── adapter_weights.pt       # lora_a weights
    │   │   └── adapter_config.json
    │   └── lora_b/
    │       ├── adapter_weights.pt       # lora_b weights
    │       └── adapter_config.json
    └── meta.json                        # includes "adapter_names": ["lora_a", "lora_b"]
```

### Merged adapter

```text
ckpt/qwen3-4b-openclaw-opd-merged-lora/
├── adapter_weights.pt                   # Averaged weights (default adapter name)
└── adapter_config.json                  # LoRA config
```

---

## 8. Script Reference

| Script | Purpose |
|--------|---------|
| `run_qwen3_4b_openclaw_opd.sh` | Token-level OPD (Megatron backend) |
| `run_qwen3_4b_openclaw_opd_topk.sh` | Top-K distillation (Megatron backend) |
| `run_qwen3_4b_openclaw_opd_topk_lora.sh` | Single LoRA + FSDP |
| `run_qwen3_4b_openclaw_opd_topk_dual_lora.sh` | Dual LoRA + FSDP (federated) |
| `run_merge_lora_adapters.sh` | Merge (weighted average) LoRA adapters |
| `run_gsm8k_eval.sh` | GSM8K evaluation (base script) |
| `run_gsm8k_easy_eval.sh` | GSM8K easy evaluation |
| `run_gsm8k_hard_eval.sh` | GSM8K hard evaluation |

---

## 9. Detailed Documentation

| Document | Description |
|----------|-------------|
| [`docs/use-cases.md`](use-cases.md) | Copy-paste use cases for the three main workflows |
| [`README.md`](../README.md) | Main OPD documentation, options A–D |
| [`README-federated-opd.md`](../README-federated-opd.md) | Dual-LoRA architecture, training step, configuration |
| [`docs/federated-dual-lora-patches.md`](federated-dual-lora-patches.md) | File-by-file code patches for dual-LoRA |
| [`docs/gsm8k-evaluation.md`](gsm8k-evaluation.md) | GSM8K easy/hard evaluation details |
| [`docs/training-errors-deep-dive.md`](training-errors-deep-dive.md) | Error root-cause analysis and fixes |
