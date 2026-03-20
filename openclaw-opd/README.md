# On-Policy Distillation (OPD) with Hindsight Hints

Online distillation for agentic tool-use: use next-turn feedback to extract hindsight hints, build a stronger teacher signal, and train the student policy on-policy.

## Core Pipeline

For each main-line turn:

1. Serve response with current policy and keep rollout log-probs.
2. When next state arrives (user reply / env feedback), judge `(response, next_state)` for hindsight usefulness.
3. Run `m` judge votes; each vote returns `+1/-1` and optional hint.
4. Keep the longest non-trivial positive hint; if none exists, drop the sample.
5. Append hint to prompt and query teacher log-probs on the original response tokens.
6. Submit training sample to SLIME.

This turns delayed feedback into token-level supervision without hand-labeled trajectories.

## Option A (Default): Token-Level OPD

Teacher signal per token:

$$A_t=\log\pi_{\text{teacher}}(a_t\mid s+\text{hint})-\log\pi_\theta(a_t\mid s)$$

Training uses PPO-style clipped policy loss with the above token-level advantage, plus KL loss:

$$\mathcal{L}=\mathcal{L}_{pg}+\beta_{KL}\mathcal{L}_{KL}$$

Default script:

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

## Option B: Top-K Logits Distillation (SDFT/SDPO-style)

Following [SDFT](https://arxiv.org/abs/2601.19897) and [SDPO](https://arxiv.org/abs/2601.20802), instead of single-token teacher targets, distill teacher top-K distribution per position. But note that we use teacher top k instead of student top k (setting in their original paper), see issue #7. e will compare teacher top-K and student top-K later.

- Teacher query: `input_top_logprobs` (`K` tokens per position).
- Stored fields: `teacher_topk_log_probs [T,K]`, `teacher_topk_indices [T,K]`.
- Loss: reverse KL over `K+1` bins (top-K + tail mass):

$$D_{KL}\left(\pi_\theta^{K+1}\|\pi_{teacher}^{K+1}\right)=\sum_{k=1}^{K+1}\pi_\theta^{(k)}\left(\log\pi_\theta^{(k)}-\log\pi_{teacher}^{(k)}\right)$$

Tail bin uses:

$$\log p_{tail}=\log\left(1-\exp(\mathrm{logsumexp}(\log p_1,\dots,\log p_K))\right)$$

### Strict Compatibility Design

Top-K is implemented as an additive extension:

- Legacy token-level OPD path is unchanged.
- `teacher_log_probs [T]` keeps original meaning for legacy path.
- Top-K uses separate fields only (`teacher_topk_log_probs`, `teacher_topk_indices`).
- Top-K loss is external custom loss (not a built-in core loss switch).
- Top-K teacher query is off by default (`--distill-topk 0`).

### How to Run Top-K

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk.sh
```

Equivalent key args:

```bash
--loss-type custom_loss \
--custom-loss-function-path topk_distillation_loss.topk_distillation_loss_function \
--distill-topk 50 \
--disable-compute-advantages-and-returns \
--entropy-coef 0.00
```

## Option C: Qwen3-4B + LoRA (FSDP, fewer GPUs)

LoRA training with FSDP backend (no Megatron conversion).

### 1. Environment setup

From the **OpenClaw-RL repo root**:

```bash
bash openclaw-opd/install_openclaw_opd_lora.sh
conda activate openclaw-opd-lora
```

Alternatively: `conda env create -f openclaw-opd/environment-openclaw-opd-lora.yml`, then run the install script with that env active.

The install script installs **both** `sglang-router` and the full **`sglang`** package (repo-pinned git ref). Slime and the rollout code require `sglang` (e.g. `sglang.srt.*`), not only `sglang-router`. If you see `ModuleNotFoundError: No module named 'sglang'`, (re)run the install script or install sglang from the same ref (see `install_openclaw_opd_lora.sh`).

### 2. Download the model (required for local path)

If you use a **local** checkpoint path, the directory must exist and contain `config.json`. For example:

```bash
export HF_CKPT=/path/to/OpenClaw-RL/models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B --local-dir "${HF_CKPT}"
```

You can also set `HF_CKPT` to a Hugging Face repo id (e.g. `Qwen/Qwen3-4B`) to load from the hub; then no download step is needed.

### 3. Run the LoRA script

**Always run with the conda env activated** so the Ray job uses the same Python (and thus sees `sglang` and other deps):

```bash
conda activate openclaw-opd-lora
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh
```

The script checks that a local `HF_CKPT` path exists and contains `config.json`; if not, it prints an error and the download command above.

- **Default paths:** `HF_CKPT` defaults to `{REPO_ROOT}/models/Qwen3-4B`; override with `export HF_CKPT=...` before running.
- **GPU layout:** `NUM_GPUS=4`, `ACTOR_GPUS=2`, `ROLLOUT_GPUS=1`, `PRM_GPUS=1`; override as needed.
- **Ref model:** The script uses `--ref-load` equal to `HF_CKPT` (no Megatron `torch_dist` needed).

### Troubleshooting

| Issue | What to do |
|-------|------------|
| `No module named 'sglang'` | Ensure the env is activated before running the script; the script uses the current shell’s `python3` for the Ray job. Reinstall with `bash openclaw-opd/install_openclaw_opd_lora.sh` if needed. |
| `HF checkpoint path not found or invalid` | Download the model (step 2) or set `HF_CKPT` to a valid local directory (with `config.json`) or a Hub repo id. |
| `Can't load the configuration of '...'` / `HFValidationError` | For a local path, ensure the directory exists and has `config.json`. The FSDP actor uses `local_files_only=True` for local dirs to avoid hub validation. |
| **`Triton is not supported on current platform, roll back to CPU`** | This is **expected and safe to ignore**. The RolloutManager Ray actor is scheduled with **no GPU** (`num_gpus=0`); it only coordinates workers. When its process imports SGLang’s FLA (Flash Linear Attention) utils, Triton sees no GPU in that process and falls back to CPU for that module. Actual model inference runs in separate SGLang engine processes that do have GPUs. |
| **`flash_attn seems to be not installed`** / **FlashAttention2 cannot be used** | The LoRA script uses `--attn-implementation sdpa` by default (PyTorch built-in), so the `flash_attn` package is not required. If you still see this error, ensure you run the latest script. To use Flash Attention 2, install `flash-attn` and set `export ATTN_IMPL=flash_attention_2` before running. |

**Note:** The top-K distillation loss in `topk_distillation_loss.py` imports `megatron.core.mpu`. The LoRA script uses `--train-backend fsdp`; the FSDP actor uses a built-in policy loss. If you need the custom top-K loss with this script, you may need Megatron-LM in `PYTHONPATH` or to use the Megatron-backed Top-K script (`run_qwen3_4b_openclaw_opd_topk.sh`) instead.

## Sending Requests (Producing OPD Samples)

The OPD server exposes an **OpenAI-compatible** chat completions API. It does not accept raw training samples; instead it creates them internally from multi-turn conversations. Each `main` turn is evaluated by the PRM when the next user message arrives, and accepted turns become OPD training samples.

Training begins once `rollout-batch-size` samples (default 4) have accumulated in the queue.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/healthz` | GET | Health check |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |

### Headers

| Header | Values | Purpose |
|--------|--------|---------|
| `X-Session-ID` | any string | Groups turns into a session |
| `X-Turn-Type` | `main` / `side` | Only `main` turns produce OPD samples |
| `X-Session-Done` | `1` / `true` | Signals end of session, triggers cleanup |

These can also be passed in the JSON body as `session_id`, `turn_type`, `session_done`.

### curl

```bash
# Health check
curl http://0.0.0.0:30000/healthz

# Turn 1
curl -X POST http://0.0.0.0:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: sess-001" \
  -H "X-Turn-Type: main" \
  -d '{
    "messages": [{"role": "user", "content": "Solve 2+2"}],
    "model": "qwen3-4b",
    "max_tokens": 2048
  }'

# Turn 2 — triggers OPD evaluation of Turn 1
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

### Python (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(api_key="dummy_key", base_url="http://0.0.0.0:30000/v1")

# Turn 1
resp1 = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Solve 2+2"}],
    max_tokens=2048,
    extra_headers={"X-Session-ID": "sess-001", "X-Turn-Type": "main"},
)

# Turn 2 — triggers OPD sample creation for Turn 1
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

### Notes

- **Multi-turn required.** OPD samples are created when the *next* message arrives (so the server can evaluate the previous turn with PRM). A single isolated request does not produce a training sample.
- **503 during weight updates.** The server briefly returns 503 while syncing new weights to SGLang — retry after a short delay.
- **PRM gating.** Not every turn produces a sample; the PRM judge must accept the turn (successful hint extraction) for it to enter the training queue.

## File Layout

```text
openclaw-opd/
├── README.md
├── environment-openclaw-opd-lora.yml        # Conda env for LoRA (Python 3.10)
├── install_openclaw_opd_lora.sh            # Install script for Qwen3-4B + LoRA
├── run_qwen3_4b_openclaw_opd.sh            # Token-level OPD (default)
├── run_qwen3_4b_openclaw_opd_topk.sh       # Top-K custom-loss path
├── run_qwen3_4b_openclaw_opd_topk_lora.sh  # Top-K + LoRA (FSDP)
├── topk_distillation_loss.py               # Reverse-KL top-K loss (external custom loss)
├── openclaw_opd_api_server.py              # Async judge + teacher query + sample submission
├── openclaw_opd_rollout.py                 # Rollout bridge to SLIME trainer
└── results/                                # Runtime records (auto-created)
```
