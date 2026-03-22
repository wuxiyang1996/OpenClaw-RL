# Federated On-Policy Distillation (Federated OPD)

Train two distinct LoRA adapters on a **single 4-GPU Ray job**, each receiving
user messages from a different API port.  Both adapters share the same base
model and LoRA hyperparameters — only the ingestion endpoint differs.

Strict serving separation is enforced: requests to port A always receive
responses from LoRA A, and requests to port B always receive responses from
LoRA B.  SGLang's native multi-LoRA serving handles the per-request routing.

## Architecture

```text
                       ┌──────────────────────────────────────────────┐
                       │            Single Ray Job – 4 GPUs           │
                       │                                              │
  Client A             │  ┌──────────────────────────────────┐        │
  ─────────────►       │  │  API Server A  :30000            │        │
  :30000               │  │    adapter_name = lora_a         │        │
                       │  │    output_queue → Queue A        │        │
                       │  └──────────┬───────────────────────┘        │
                       │             │                                │
  Client B             │  ┌──────────┴───────────────────────┐        │
  ─────────────►       │  │  API Server B  :30001            │        │
  :30001               │  │    adapter_name = lora_b         │        │
                       │  │    output_queue → Queue B        │        │
                       │  └──────────┬───────────────────────┘        │
                       │             │                                │
                       │  ┌──────────▼───────────────────────────┐    │
                       │  │  SGLang Engine (1 GPU)                │    │
                       │  │    Base model + LoRA A + LoRA B       │    │
                       │  │    (native multi-LoRA, max_loras=2)   │    │
                       │  └──────────┬───────────────────────────┘    │
                       │             │                                │
                       │  ┌──────────▼───────────────────────────┐    │
                       │  │  FSDP Actor (2 GPUs)                  │    │
                       │  │    Base model (frozen)                │    │
                       │  │    + LoRA A params (PEFT adapter)     │    │
                       │  │    + LoRA B params (PEFT adapter)     │    │
                       │  │    → single optimizer for both        │    │
                       │  └──────────────────────────────────────┘    │
                       │                                              │
                       │  ┌──────────────────────────────────────┐    │
                       │  │  PRM / Teacher (1 GPU)                │    │
                       │  └──────────────────────────────────────┘    │
                       └──────────────────────────────────────────────┘
```

## Training Step (every step processes both adapters)

```python
optimizer.zero_grad()

model.set_adapter("lora_a")
loss_a = forward(data_from_queue_a)
loss_a.backward()                    # gradients on lora_a params

model.set_adapter("lora_b")
loss_b = forward(data_from_queue_b)
loss_b.backward()                    # gradients on lora_b params

optimizer.step()                     # updates both adapter param sets

sync_adapter_to_sglang("lora_a")     # save adapter → SGLang reloads
sync_adapter_to_sglang("lora_b")
```

## Data Isolation

| Message destination | LoRA A updates | LoRA B updates |
|---------------------|:--------------:|:--------------:|
| `:30000` (Server A) | Yes            | No             |
| `:30001` (Server B) | No             | Yes            |

Messages to port A only train LoRA A; messages to port B only train LoRA B.
There is no cross-adapter data sharing at runtime.

## GPU Requirements

This setup uses **the same 4 GPUs** as the single-LoRA baseline:

| Component       | GPUs | Notes                                |
|-----------------|------|--------------------------------------|
| FSDP Actor      | 2    | Hosts both LoRA adapters via PEFT    |
| SGLang Rollout  | 1    | Multi-LoRA serving (both adapters)   |
| PRM / Teacher   | 1    | Shared across both adapters          |
| **Total**       | **4** |                                     |

Memory overhead for the second adapter is ~100–200 MB (two rank-16 adapters
are negligible compared to the base model).

## Quick Start

### 1. Environment setup

```bash
bash openclaw-opd/install_openclaw_opd_lora.sh
conda activate openclaw-opd-lora
```

### 2. Download the model

```bash
export HF_CKPT=/path/to/OpenClaw-RL/models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B --local-dir "${HF_CKPT}"
```

### 3. Run the dual-LoRA script

```bash
conda activate openclaw-opd-lora
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

A **single Ray job** is launched on 4 GPUs.  Both API servers start
automatically on their respective ports.

### 4. Override settings

```bash
PORT_A=30000 PORT_B=30001 \
LORA_RANK=32 LORA_ALPHA=64 \
ADAPTER_NAME_A=client_x ADAPTER_NAME_B=client_y \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

## Configuration Reference

### Shared

| Variable      | Default            | Description                     |
|---------------|--------------------|---------------------------------|
| `HF_CKPT`    | `models/Qwen3-4B`  | Base model checkpoint           |
| `LORA_RANK`  | 16                 | LoRA rank (both adapters)       |
| `LORA_ALPHA` | 32                 | LoRA alpha (both adapters)      |
| `NUM_GPUS`   | 4                  | Total GPUs for the Ray job      |
| `ACTOR_GPUS` | 2                  | Actor GPUs (hosts both LoRAs)   |
| `TP`         | 1                  | Tensor parallelism degree       |
| `PRM_M`      | 3                  | PRM judge vote count            |

### Per-adapter

| Variable         | Default   | Description                       |
|------------------|-----------|-----------------------------------|
| `PORT_A`         | 30000     | API port for adapter A            |
| `PORT_B`         | 30001     | API port for adapter B            |
| `ADAPTER_NAME_A` | `lora_a`  | PEFT adapter name for A           |
| `ADAPTER_NAME_B` | `lora_b`  | PEFT adapter name for B           |

## Sending Requests

Both servers expose the same OpenAI-compatible API.  The only difference is
the port — routing to the correct LoRA adapter is handled automatically.

```bash
# Adapter A
curl -X POST http://0.0.0.0:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: sess-a-001" \
  -H "X-Turn-Type: main" \
  -d '{"messages": [{"role": "user", "content": "Hello from client A"}], "model": "qwen3-4b"}'

# Adapter B
curl -X POST http://0.0.0.0:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: sess-b-001" \
  -H "X-Turn-Type: main" \
  -d '{"messages": [{"role": "user", "content": "Hello from client B"}], "model": "qwen3-4b"}'
```

### Python (OpenAI client)

```python
from openai import OpenAI

client_a = OpenAI(api_key="dummy", base_url="http://0.0.0.0:30000/v1")
client_b = OpenAI(api_key="dummy", base_url="http://0.0.0.0:30001/v1")

resp_a = client_a.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Hello from client A"}],
    extra_headers={"X-Session-ID": "sess-a-001", "X-Turn-Type": "main"},
)

resp_b = client_b.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Hello from client B"}],
    extra_headers={"X-Session-ID": "sess-b-001", "X-Turn-Type": "main"},
)
```

## Key Properties

- **4 GPUs total** — identical hardware to the single-LoRA setup
- **Both always serving** — SGLang multi-LoRA handles concurrent requests
  for both adapters; strict port → adapter mapping
- **Both train every step** — no time-slot alternation; each step does two
  sequential forward/backward passes then one `optimizer.step()`
- **Memory overhead ~100–200 MB** — two rank-16 adapters are negligible
  vs. the base model
- **Training throughput** — each step takes roughly 2× a single-LoRA step
  (two forward/backward passes), but both adapters make progress every step

## Resume from a Previous LoRA Checkpoint

Start dual-LoRA training from a previously trained single-LoRA checkpoint.
The checkpoint is automatically duplicated into both `lora_a` and `lora_b`:

```bash
# From a single-LoRA checkpoint (both adapters start identical)
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-lora \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh

# From a merged adapter (output of merge_lora_adapters.py)
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-merged-lora \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh

# Resume from a previous dual-LoRA checkpoint (each adapter loads its own weights)
LOAD_CKPT=/workspace/OpenClaw-RL/ckpt/qwen3-4b-openclaw-opd-topk-dual-lora \
  bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh
```

When `LOAD_CKPT` is set, the script adds `--load <path> --no-load-optim --no-load-rng --finetune`
so the optimizer and learning rate schedule start fresh.

## Merging Adapters After Training

Average the two LoRA adapters into a single adapter:

```bash
# Equal average (default)
bash openclaw-opd/run_merge_lora_adapters.sh

# Weighted: 70% lora_a + 30% lora_b
WEIGHTS="0.7 0.3" bash openclaw-opd/run_merge_lora_adapters.sh

# Custom paths
CKPT_DIR=/path/to/dual-lora-ckpt OUTPUT=/path/to/merged \
  bash openclaw-opd/run_merge_lora_adapters.sh
```

The merged adapter can be:
- Evaluated with `run_gsm8k_eval.sh` (as a regular single-LoRA checkpoint)
- Used as the starting point for another round of dual-LoRA training (via `LOAD_CKPT`)
- Merged into the base model with `slime/tools/merge_lora_adapter.py` for deployment

### Typical Workflow

```text
Single-LoRA training → LOAD_CKPT → Dual-LoRA training → merge adapters
                                          ↓
                          LOAD_CKPT → another round of dual-LoRA
```

## File Layout

```text
openclaw-opd/
├── README.md                                    # Main OPD documentation
├── README-federated-opd.md                      # This file
├── run_qwen3_4b_openclaw_opd_topk_lora.sh      # Single-LoRA baseline
├── run_qwen3_4b_openclaw_opd_topk_dual_lora.sh # Dual-LoRA (federated)
├── merge_lora_adapters.py                       # Weighted average of LoRA adapters
├── run_merge_lora_adapters.sh                   # Shell wrapper for merge
├── openclaw_opd_api_server.py                   # API server (adapter-aware)
├── openclaw_opd_rollout.py                      # Rollout bridge (dual queues)
├── topk_distillation_loss.py                    # Top-K distillation loss
└── results/
    └── qwen3_4b_dual_lora_record.jsonl          # Combined records

slime/
├── train_async.py                               # Original async training loop
├── train_async_federated.py                     # Federated async training loop
└── slime/backends/fsdp_utils/
    ├── actor.py                                 # Multi-adapter actor support
    ├── lora_utils.py                            # Multi-adapter LoRA utilities
    ├── checkpoint.py                            # Multi-adapter checkpointing + resume
    └── arguments.py                             # num_lora_adapters, adapter names
```

## Limitations & Future Work

- **No cross-adapter training.** Each LoRA trains only on its own port's
  traffic.
- **No automatic periodic FedAvg.** Merging is a manual post-training step.
  A periodic in-training FedAvg could be added to the training loop.
- **Scaling beyond 2.** The architecture generalizes to N adapters (with
  `--num-lora-adapters N`), limited mainly by GPU memory and SGLang's
  `max_loras` setting.
