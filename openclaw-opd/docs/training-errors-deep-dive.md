# Training Errors Deep-Dive: OpenClaw-RL OPD + LoRA (FSDP)

This document provides a detailed root-cause analysis of the errors encountered when
running the Qwen3-4B + LoRA training pipeline via
`run_qwen3_4b_openclaw_opd_topk_lora.sh`, along with the fixes applied and guidance
for future troubleshooting.

---

## Environment Summary

| Component         | Value                                |
|-------------------|--------------------------------------|
| GPU               | NVIDIA A100-SXM4-80GB               |
| PyTorch           | 2.9.1+cu128                          |
| Python            | 3.10 (conda env `openclaw-opd-lora`) |
| Training Backend  | FSDP (Fully Sharded Data Parallel)   |
| Model             | Qwen3-4B (Qwen/Qwen3-4B)            |
| Adapter           | LoRA (rank 16, alpha 32)             |
| Orchestrator      | Ray (job submission)                 |
| Inference Engine  | SGLang                               |

---

## Error #1: Model Configuration Not Found

### Symptom

```
OSError: Can't load the configuration of '/workspace/OpenClaw-RL/models/Qwen3-4B'.
If you were trying to load it from 'https://huggingface.co/models', make sure you
don't have a local directory with the same name. Otherwise, make sure
'/workspace/OpenClaw-RL/models/Qwen3-4B' is the correct path to a directory
containing a config.json file
```

Preceded by a `HFValidationError`:
```
huggingface_hub.errors.HFValidationError: Repo id must be in the form
'repo_name' or 'namespace/repo_name': '/workspace/OpenClaw-RL/models/Qwen3-4B'.
Use `repo_type` argument if needed.
```

### Root Cause

The training script was passed `--hf-checkpoint /workspace/OpenClaw-RL/models/Qwen3-4B`,
but the directory either did not exist or was empty (no `config.json`).

When `transformers.AutoConfig.from_pretrained()` receives a local path, it first
checks if the path is a valid local directory with model files. If that fails, it
falls back to treating it as a HuggingFace Hub `repo_id`. But an absolute path like
`/workspace/...` is not a valid repo_id format, causing the `HFValidationError`.
This then cascades into the `OSError`.

### Call Chain

```
train_async.py:22  train()
  → placement_group.py:160  create_training_models()
    → ray.get(FSDPTrainRayActor.init())
      → actor.py:82  AutoConfig.from_pretrained(_hf_path, ...)
        → configuration_utils.py:721  _get_config_dict()
          → hub.py:322  cached_file()
            → _validators.py:154  validate_repo_id()  ← HFValidationError
          → configuration_utils.py:744  raise OSError(...)
```

### Fix Applied

The model was downloaded to the expected path:

```bash
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir /workspace/OpenClaw-RL/models/Qwen3-4B
```

### Code Defense (already in place)

The run script (`run_qwen3_4b_openclaw_opd_topk_lora.sh`) already includes a
pre-flight check at lines 44-51:

```bash
if [[ "${HF_CKPT}" == /* || ... ]]; then
  if [[ ! -d "${HF_CKPT}" || ! -f "${HF_CKPT}/config.json" ]]; then
    echo "ERROR: HF checkpoint path not found or invalid: ${HF_CKPT}"
    ...
    exit 1
  fi
fi
```

The FSDP actor (`actor.py`) also uses `local_files_only=True` when the path is a
local directory (lines 78-84), preventing hub API calls for local paths:

```python
_local_only = os.path.isdir(_hf_path)
self.hf_config = AutoConfig.from_pretrained(
    _hf_path, trust_remote_code=True, local_files_only=_local_only
)
```

### Prevention

- Always download the model before first run.
- Override `HF_CKPT` to point to a Hub repo id (e.g. `Qwen/Qwen3-4B`) to
  skip local download entirely.

---

## Error #2: FlashAttention2 Not Installed

### Symptom

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to
the following error: the package flash_attn seems to be not installed. Please
refer to the documentation of
https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
to install Flash Attention 2.
```

### Root Cause

Two defaults were set to `flash_attention_2`:

1. **`arguments.py`** (`FSDPArgs` dataclass):
   ```python
   attn_implementation: str = "flash_attention_2"  # OLD default
   ```

2. **`run_qwen3_4b_openclaw_opd_topk_lora.sh`** (env var):
   ```bash
   export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"  # OLD default
   ```

This value flows through as `--attn-implementation flash_attention_2` and is used in
`actor.py` line 98:

```python
model = self.get_model_cls().from_pretrained(
    _hf_path,
    trust_remote_code=True,
    attn_implementation=self.args.attn_implementation,  # "flash_attention_2"
    local_files_only=_local_only,
)
```

`AutoModelForCausalLM.from_pretrained()` passes this to the Qwen3 model constructor,
which calls `_check_and_adjust_attn_implementation()`. This function checks if the
`flash_attn` Python package is importable. Since it is not installed in the
`openclaw-opd-lora` conda environment, the `ImportError` is raised.

### Call Chain

```
train_async.py:22  train()
  → placement_group.py:160  create_training_models()
    → ray.get(FSDPTrainRayActor.init())
      → actor.py:98  get_model_cls().from_pretrained(..., attn_implementation="flash_attention_2")
        → auto_factory.py:604  model_class.from_pretrained()
          → modeling_utils.py:4971  cls(config, ...)
            → modeling_qwen3.py:435  super().__init__(config)
              → modeling_utils.py:2076  _check_and_adjust_attn_implementation()
                → modeling_utils.py:2714  _flash_attn_2_can_dispatch()
                  → modeling_utils.py:2422  raise ImportError(...)
```

### Why flash_attn Was Missing

The `install_openclaw_opd_lora.sh` install script does not explicitly install
`flash-attn`. It attempts to install `ring_flash_attn` (which depends on `flash_attn`),
but this is wrapped in a `|| echo "WARN: ..."` fallback and fails silently.

Building `flash-attn` from source requires CUDA toolkit headers and a C++ compiler,
and typically takes 20-40 minutes. There is no pre-built wheel for the installed
PyTorch version (2.9.1+cu128), so the install silently failed.

### Fix Applied

Changed the default attention implementation from `flash_attention_2` to `sdpa`
(PyTorch's built-in Scaled Dot Product Attention) in two files:

**`slime/slime/backends/fsdp_utils/arguments.py`:**
```python
attn_implementation: str = "sdpa"  # was "flash_attention_2"
```

**`openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh`:**
```bash
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"  # was "flash_attention_2"
```

### Why `sdpa` is the Right Default

| Aspect | `flash_attention_2` | `sdpa` | `eager` |
|--------|---------------------|--------|---------|
| Requires extra package | Yes (`flash-attn`) | No (built into PyTorch) | No |
| A100 performance | Best (~10-20% faster) | Very good | Slowest |
| Memory efficiency | Best (tiling) | Good (fused kernels) | Worst |
| Compatibility | Requires matching CUDA/torch versions | Always works | Always works |

`sdpa` provides a good balance: no extra dependencies, solid performance on A100,
and automatic kernel selection (it can use Flash Attention kernels internally if
conditions are met).

### Optionally Enabling Flash Attention 2

For maximum performance on A100/H100, install `flash-attn` and override the env var:

```bash
# Install (may take 20-40 min to compile)
pip install flash-attn --no-build-isolation

# Then run with flash_attention_2
export ATTN_IMPL=flash_attention_2
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh
```

---

## Attention Implementation Options Reference

The `--attn-implementation` argument controls how self-attention is computed during
training. This is a HuggingFace Transformers parameter passed directly to
`from_pretrained()`.

### Available Backends

| Backend | Value | Package Required | Notes |
|---------|-------|-----------------|-------|
| SDPA | `sdpa` | None (PyTorch built-in) | **Default.** Uses `torch.nn.functional.scaled_dot_product_attention`. Auto-selects the best available kernel (Flash, Memory-Efficient, or Math). |
| Flash Attention 2 | `flash_attention_2` | `flash-attn>=2.0` | Explicit Flash Attention 2 from Tri Dao. Best throughput on Ampere/Hopper GPUs. Requires matching CUDA/torch versions. |
| Eager | `eager` | None | Manual attention implementation. Materializes full attention matrix. Useful for debugging. |

### How It Flows Through the Code

```
Shell script (ATTN_IMPL env var)
  → CLI arg (--attn-implementation)
    → arguments.py (FSDPArgs.attn_implementation)
      → actor.py line 98-103 (from_pretrained(..., attn_implementation=...))
        → actor.py line 835-839 (ref model also uses same implementation)
```

---

## Error #3: `sgl_kernel` Architecture Mismatch (SM100 vs SM80)

### Symptom

```
ImportError:
[sgl_kernel] CRITICAL: Could not load any common_ops library!

Attempted locations:
1. Architecture-specific pattern: .../sgl_kernel/sm100/common_ops.* - found files:
   ['.../sgl_kernel/sm100/common_ops.abi3.so']
2. Fallback pattern: .../sgl_kernel/common_ops.* - found files: []
3. Standard Python import: common_ops - failed

GPU Info:
- Compute capability: 80
- Expected variant: SM80 (precise math for compatibility)

Error details from previous import attempts:
- ImportError: libnuma.so.1: cannot open shared object file: No such file or directory
- ModuleNotFoundError: No module named 'common_ops'
```

### Root Cause

Two issues combine to cause this failure:

1. **`sgl_kernel` version ships only SM100 binaries.** The installed `sgl_kernel`
   0.3.21 only contains pre-built binaries for SM100 (NVIDIA Blackwell B200/B100),
   located at `sgl_kernel/sm100/common_ops.abi3.so`. The GPU on this machine is an
   **A100 (compute capability 80 / SM80)**, so the SM100 binary is incompatible.
   There is no `sm80/` directory in the package.

2. **`libnuma.so.1` is missing.** The system library `libnuma` is not installed in
   the container. Even the fallback import paths fail because the shared object
   `libnuma.so.1` (a dependency of the compiled kernel) cannot be loaded.

### Call Chain

```
train_async.py:22  train()
  → placement_group.py:172  actor_model.set_rollout_manager(rollout_manager)
    → ray.get(RolloutManager.__init__())
      → rollout.py:82  init_rollout_engines()
        → rollout.py:875  ray.get(SGLangEngine.init())
          → sglang_engine.py:196  launch_server_process(ServerArgs(...))
            → server_args.py:733  __post_init__()
              → server_args.py:5109  use_mla_backend()
                → model_config.py:27  from sglang.srt.layers.quantization import ...
                  → fp8_kernel.py:50  from sgl_kernel import sgl_per_token_quant_fp8
                    → sgl_kernel/__init__.py:5  _load_architecture_specific_ops()
                      → load_utils.py:188  raise ImportError(...)
```

### Fix

Two steps are needed:

```bash
# 1. Install the missing system library
apt-get update && apt-get install -y libnuma-dev

# 2. Install sgl_kernel 0.3.20 — the version sglang expects, with SM80 support
pip install sgl_kernel==0.3.20
```

**Important:** Do NOT use `sgl_kernel==0.3.6.post2` — it has an ABI mismatch with
PyTorch 2.9.1 (`undefined symbol: _ZNK3c106SymInt6sym_neERKS0_`). And do NOT use
`sgl_kernel==0.3.21` (latest) — it only ships SM100 binaries.

`sgl_kernel==0.3.20` is the correct version because:
- It matches what the installed `sglang` commit declares as its dependency.
- It includes SM80 support (via fallback loading, not a dedicated `sm80/` directory).
- It is ABI-compatible with PyTorch 2.9.1+cu128.

To verify the fix:

```bash
python -c "import sgl_kernel; print('OK, version:', sgl_kernel.version.__version__)"
```

### Why This Happened

The `install_openclaw_opd_lora.sh` script installs `sglang` from a pinned git
commit, which pulls in `sgl_kernel` as a transitive dependency. Running
`pip install --upgrade sgl_kernel` later upgraded it to 0.3.21, which only ships
SM100 (Blackwell) pre-built wheels. The OpenClaw-RL project was developed and tested
on A100s (SM80), but the latest `sgl_kernel` releases no longer bundle SM80 binaries
in the default wheel.

### Prevention

- Pin `sgl_kernel` to a known-good version in the install script.
- Always verify GPU compute capability matches the installed kernel binaries.
- Ensure `libnuma-dev` is installed in the base container image.

---

## Error #4: `sgl_kernel` 0.3.6.post2 ABI Mismatch

### Symptom

```
ImportError: .../sgl_kernel/common_ops.abi3.so: undefined symbol:
_ZNK3c106SymInt6sym_neERKS0_
```

### Root Cause

`sgl_kernel==0.3.6.post2` was compiled against an older PyTorch version. The
symbol `c10::SymInt::sym_ne` (demangled from `_ZNK3c106SymInt6sym_neERKS0_`) was
removed or changed in PyTorch 2.9.1. The `.abi3.so` shared object tries to link
against this symbol at import time and fails.

### Fix

Install the version that matches the sglang dependency and is ABI-compatible:

```bash
pip install sgl_kernel==0.3.20
```

---

## Warning: Deprecated `tool_call_parser` Value

### Symptom

```
WARNING server_args.py:833: The tool_call_parser 'qwen25' is deprecated.
Please use 'qwen' instead.
```

### Fix Applied

Changed the default in `run_qwen3_4b_openclaw_opd_topk_lora.sh`:

```bash
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"  # was "qwen25"
```

---

## Pip Dependency Warnings Assessment

After installing `sgl_kernel==0.3.20`, pip reports these remaining conflicts.
All are safe to ignore for this training pipeline:

| Package | Expected | Installed | Risk |
|---------|----------|-----------|------|
| `grpcio-tools` | 1.75.1 | not installed | Only needed for gRPC stub generation; slime uses in-process SGLang |
| `grpcio` | 1.75.1 | 1.78.0 | Minor version bump, backwards-compatible |
| `grpcio-health-checking` | 1.75.1 | 1.78.0 | Same as grpcio |
| `grpcio-reflection` | 1.75.1 | 1.78.0 | Same as grpcio |
| `flashinfer_cubin` | 0.5.3 | 0.6.3 | Larger gap but imports successfully; was already installed before any issues |
| `flashinfer_python` | 0.5.3 | 0.6.3 | Same as flashinfer_cubin |
| `nvidia-cutlass-dsl` | 4.2.1 | 4.3.5 | Minor version bump, backwards-compatible |

---

## Error #5: sglang Runtime Version Gate Rejects `sgl_kernel==0.3.20`

### Symptom

```
Exception: sgl-kernel is installed with version 0.3.20, which is less than
the minimum required version 0.3.21. Please reinstall the latest version
with `pip install sgl-kernel --force-reinstall`
```

### Root Cause

The sglang code has a hard runtime version check in `engine.py` line 856:

```python
if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
    if _is_cuda:
        assert_pkg_version("sgl-kernel", "0.3.21", ...)
```

This enforces `sgl_kernel >= 0.3.21`, but 0.3.21 only ships SM100 (Blackwell)
binaries. On A100 (SM80) we must use 0.3.20. The check has an official bypass
via the `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` env var.

### Fix Applied

Added to `run_qwen3_4b_openclaw_opd_topk_lora.sh`:

```bash
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
```

Also added to the `RUNTIME_ENV_JSON` so the env var propagates to Ray worker
processes (where SGLangEngine runs):

```json
"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1"
```

---

## Full Error Timeline

| Run | Timestamp | Error | Status |
|-----|-----------|-------|--------|
| 1 | 20:21:14 | `OSError: Can't load the configuration` — model not downloaded | Fixed: model downloaded |
| 2 | 20:27:15 | `ImportError: FlashAttention2 ... flash_attn not installed` | Fixed: default changed to `sdpa` |
| 3 | 20:33:11 | Same FlashAttention2 error (re-ran without code fix) | Fixed: default changed to `sdpa` |
| 4 | 20:40:13 | `ImportError: sgl_kernel could not load common_ops` — SM100 binary on SM80 GPU + missing `libnuma.so.1` | Fixed: `libnuma-dev` installed + `sgl_kernel==0.3.20` |
| 5 | 21:05:58 | `ImportError: undefined symbol _ZNK3c106SymInt6sym_neERKS0_` — sgl_kernel 0.3.6.post2 ABI mismatch | Fixed: `sgl_kernel==0.3.20` |
| 6 | 21:12:44 | `Exception: sgl-kernel 0.3.20 < minimum 0.3.21` — sglang runtime version gate | Fixed: `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` |

---

## Package Dependency Audit

A comprehensive audit of all Python imports across the OpenClaw-RL codebase was
performed against the installed packages in the `openclaw-opd-lora` conda environment.

### Methodology

1. Extracted all `import X` and `from X import` statements from every `.py` file
   in `openclaw-opd/` and `slime/`.
2. Categorized imports as standard library, third-party, or internal.
3. Tested every third-party import with `python -c "import X"` inside the conda env.
4. Traced the actual FSDP LoRA training code path to identify which packages are
   critical vs. optional.

### Results: 3 Missing Packages

| Package | Import Status | Required for FSDP LoRA? | Notes |
|---------|--------------|------------------------|-------|
| `flash_attn` | NOT installed | No | No pre-built wheel for torch 2.9.1+cu128. NVCC 12.4 on system cannot compile for cu128 torch. SDPA is used instead. |
| `ring_flash_attn` | Installed (0.1.8) but BROKEN | No | Fails to import because it depends on `flash_attn`. Used only for context parallelism in the Megatron backend. |
| `megatron` (Megatron-LM) | NOT installed | No | Only used by the Megatron training backend and `topk_distillation_loss.py`. The FSDP backend has its own loss computation. |

### Why `flash_attn` Cannot Be Installed

The environment has a version mismatch that prevents compilation:

| Component | Version |
|-----------|---------|
| PyTorch | 2.9.1+**cu128** (built against CUDA 12.8) |
| System NVCC | 12.4 (V12.4.131) |
| flash-attn latest | 2.8.3 (pre-built wheels only up to torch 2.8) |

The flash-attn build system uses the local `nvcc` to compile CUDA kernels, but the
local NVCC (12.4) does not match the PyTorch CUDA version (12.8). No pre-built
wheel exists on PyPI or GitHub releases for the torch 2.9 + CUDA 12.8 combination.

**Impact:** None for the default FSDP LoRA training path, which uses `sdpa`. PyTorch's
SDPA internally dispatches to Flash Attention kernels when possible (via `FlashAttentionBackend`
in `torch.nn.attention`), so performance is comparable.

### Why `megatron` Is Not Needed for FSDP

The training script passes `--loss-type custom_loss --custom-loss-function-path
topk_distillation_loss.topk_distillation_loss_function`, but the FSDP backend
**does not use the custom loss function**. The code path is:

```
FSDP actor._train_step()
  → compute_policy_loss()          ← from slime.utils.ppo_utils (no megatron import)
  → loss = pg_loss - entropy_coef * entropy_loss

Megatron actor.train()             ← DIFFERENT code path, NOT used with --train-backend fsdp
  → loss.py: load_function(args.custom_loss_function_path)
    → topk_distillation_loss.py    ← imports megatron.core.mpu
```

The `--loss-type` and `--custom-loss-function-path` arguments are parsed and stored
in `args` but **never referenced** by the FSDP backend's `_train_step` method.

**Important caveat:** This means the FSDP LoRA training uses PPO-style clipped policy
loss, NOT the Top-K logits distillation loss that the script arguments suggest.
If Top-K distillation is required with FSDP, a custom loss implementation
without megatron dependencies would need to be added to the FSDP backend.

### 37 Packages Verified Working

All other third-party packages import successfully in the conda environment:

`torch`, `transformers`, `accelerate`, `peft`, `ray`, `sglang`, `sglang_router`,
`numpy`, `tqdm`, `wandb`, `yaml`, `httpx`, `fastapi`, `uvicorn`, `datasets`,
`omegaconf`, `blobfile`, `pillow`, `pylatexenc`, `tensorboard`, `triton`,
`packaging`, `safetensors`, `pydantic`, `qwen_vl_utils`, `einops`, `typer`,
`sympy`, `starlette`, `openai`, `requests`, `aiohttp`, `torch_memory_saver`,
`pybase64`, `sentencepiece`, `tiktoken`, `scipy`

### FSDP Training Path: Full Import Verification

All 15 critical module imports for the FSDP LoRA training path were verified:

| Module | Status |
|--------|--------|
| `transformers.AutoConfig` / `AutoModelForCausalLM` / `AutoTokenizer` | OK |
| `torch.nn.functional.scaled_dot_product_attention` (SDPA) | OK |
| `peft.LoraConfig` / `get_peft_model` | OK |
| `accelerate.init_empty_weights` | OK |
| `torch.distributed.fsdp.fully_shard` (FSDP2) | OK |
| `sglang.srt.server_args.ServerArgs` | OK |
| `openclaw_opd_api_server.generate` / `reward_func` | OK |
| `openclaw_opd_rollout.generate_rollout_openclaw_opd` | OK |
| `slime.backends.fsdp_utils.actor.FSDPTrainRayActor` | OK |
| `slime.backends.fsdp_utils.checkpoint` | OK |
| `slime.backends.fsdp_utils.lr_scheduler` | OK |
| `slime.backends.fsdp_utils.data_packing` | OK |
| `slime.backends.fsdp_utils.update_weight_utils` | OK |
| `slime.backends.fsdp_utils.lora_utils` | OK |
| Qwen3-4B model config + tokenizer load | OK |

### Newly Installed

| Package | Version | Reason |
|---------|---------|--------|
| `pytest` | 9.0.2 | Required for running unit tests (`tests/`) |

---

## Quick Reference: Running the Training

```bash
# 1. Activate the conda environment
conda activate openclaw-opd-lora

# 2. Download model (first time only)
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir /workspace/OpenClaw-RL/models/Qwen3-4B

# 3. Run training (uses sdpa by default, no flash-attn needed)
cd /workspace/OpenClaw-RL/slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh

# Optional: override for flash attention 2 (if flash-attn is installed)
# ATTN_IMPL=flash_attention_2 bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh
```

---

## Key Files Modified

### Initial setup / environment fixes

| File | Change |
|------|--------|
| `slime/slime/backends/fsdp_utils/arguments.py` | Default `attn_implementation` changed from `flash_attention_2` to `sdpa`; added `num_lora_adapters` and `lora_adapter_names` fields |
| `openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh` | Default `ATTN_IMPL` → `sdpa`; `TOOL_CALL_PARSER` → `qwen`; added `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` to both script and `RUNTIME_ENV_JSON` |

### Federated dual-LoRA feature

See [`federated-dual-lora-patches.md`](federated-dual-lora-patches.md) for
full details on each patch. Summary of changed files:

| File | Change |
|------|--------|
| `slime/slime/backends/fsdp_utils/lora_utils.py` | Multi-adapter LoRA utilities: `add_lora_adapter`, `set_all_lora_requires_grad`, adapter-specific save/load |
| `slime/slime/backends/fsdp_utils/actor.py` | Multi-adapter init, training (`_train_core_multi_adapter`), adapter-only weight sync, multi-adapter checkpointing |
| `slime/slime/backends/fsdp_utils/checkpoint.py` | `save_multi_lora()` for per-adapter checkpoint saving |
| `slime/slime/backends/sglang_utils/sglang_engine.py` | `load_lora_adapter` / `unload_lora_adapter` methods; `max_loras` in server args |
| `slime/slime/utils/types.py` | `adapter_name` field on `Sample` |
| `slime/slime/ray/rollout.py` | Propagate `adapter_names` through data pipeline and DP splitting |
| `openclaw-opd/openclaw_opd_api_server.py` | Adapter-aware request routing via `"model": adapter_name` |
| `openclaw-opd/openclaw_opd_rollout.py` | Dual API servers with separate queues; federated queue draining |
| `slime/train_async_federated.py` | New entry point for federated training |
| `openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh` | Single 4-GPU Ray job with `PORT_A`/`PORT_B` and `--num-lora-adapters 2` |
| `openclaw-opd/README-federated-opd.md` | User-facing documentation for the dual-LoRA feature |
