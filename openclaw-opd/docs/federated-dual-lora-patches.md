# Federated Dual-LoRA: Code Patches & Design Notes

This document details every code change made to support training two distinct
LoRA adapters concurrently on a single 4-GPU Ray job with strict serving
separation (port A → LoRA A, port B → LoRA B).

For user-facing usage instructions, see
[`README-federated-opd.md`](../README-federated-opd.md).

---

## Design Overview

The standard single-LoRA pipeline uses 4 GPUs:

```
Actor (2 GPU, FSDP)  ─→  SGLang rollout (1 GPU)  ─→  PRM (1 GPU)
```

The federated design keeps the **same 4-GPU layout** but multiplexes two LoRA
adapters through it:

- **SGLang**: started with `max_loras=2`, serves both adapters via native
  multi-LoRA. Requests include `"model": "lora_a"` or `"model": "lora_b"` to
  route to the correct adapter.
- **Actor**: uses PEFT's multi-adapter API (`add_adapter`, `set_adapter`) to
  keep both adapters in memory. A single `optimizer.step()` updates both.
- **Rollout**: two `OpenClawOPDAPIServer` instances on different ports, each
  tagging samples with its adapter name.

---

## File-by-File Patch Summary

### 1. `slime/slime/backends/fsdp_utils/lora_utils.py`

**Purpose:** Multi-adapter LoRA utilities.

| Change | Details |
|--------|---------|
| Extracted `_build_lora_config(args)` | Shared config builder so `apply_lora` and `add_lora_adapter` reuse the same config. |
| `apply_lora(model, args, adapter_name="default")` | New `adapter_name` parameter passed to `get_peft_model()`. |
| Added `add_lora_adapter(model, args, adapter_name)` | Calls `model.add_adapter(name, config)` on an existing PeftModel. |
| Added `set_all_lora_requires_grad(model, requires_grad=True)` | Forces `requires_grad` on **all** LoRA parameters, overriding PEFT's per-adapter toggle. This is critical: PEFT's `set_adapter()` disables `requires_grad` on inactive adapters, which would prevent the optimizer from seeing their gradients. |
| `save_lora_checkpoint(model, path, adapter_name=None)` | When `adapter_name` is given, filters saved keys to only those containing `.{adapter_name}.` in the key name. |
| `load_lora_checkpoint(model, path, adapter_name=None)` | Same adapter-name filtering when loading. |

**Key design decision:** `set_all_lora_requires_grad` exists because PEFT's
`set_adapter("lora_a")` calls `param.requires_grad_(False)` on `lora_b`'s
parameters. Since both adapters share a single optimizer, we need all LoRA
params to remain trainable. The forward graph naturally routes gradients only
to the active adapter (inactive LoRA matrices are bypassed in forward), so
keeping `requires_grad=True` on all params is safe — inactive params simply
get zero gradient.

### 2. `slime/slime/backends/fsdp_utils/arguments.py`

**Purpose:** New CLI arguments for multi-adapter mode.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_lora_adapters` | `int` | `1` | Number of LoRA adapters. `>1` enables multi-adapter mode. |
| `lora_adapter_names` | `str \| None` | `None` | Comma-separated adapter names, e.g. `"lora_a,lora_b"`. Auto-generated if not provided. |

### 3. `slime/slime/backends/fsdp_utils/actor.py` — `FSDPTrainRayActor`

**Purpose:** Multi-adapter initialization, training, weight sync, checkpointing.

#### `init()`

When `num_lora_adapters > 1`:

```python
# 1. Create PeftModel with first adapter
model = apply_lora(model, args, adapter_name="lora_a")

# 2. Add second adapter (same config, different name)
add_lora_adapter(model, args, adapter_name="lora_b")

# 3. Force all LoRA params trainable (override PEFT toggle)
set_all_lora_requires_grad(model, True)

# 4. FSDP wrap (operates on both adapters)
model = apply_fsdp2(model, ...)

# 5. Optimizer receives all LoRA params from both adapters
optim_params = [p for p in model.parameters() if p.requires_grad]
```

The optimizer receives ~2× the LoRA parameters (both adapter sets), but since
rank-16 LoRA is tiny compared to the base model, memory overhead is negligible
(~100–200 MB for Qwen3-4B).

#### `_train_core()` routing

```python
def _train_core(self, rollout_id, rollout_data):
    if self._num_lora_adapters > 1 and "adapter_names" in rollout_data:
        self._train_core_multi_adapter(rollout_id, rollout_data)
        return
    # ... original single-adapter path unchanged
```

#### `_train_core_multi_adapter()`

The multi-adapter training step:

1. **Split data** by adapter name → `{adapter_name: rollout_subset}`
2. **`optimizer.zero_grad()`**
3. For each adapter in order:
   - `model.set_adapter(adapter_name)`
   - `set_all_lora_requires_grad(model, True)` (undo PEFT toggle)
   - Compute log-probs (actor + optional ref model)
   - Forward → backward → gradients accumulate on active adapter's params
4. **`clip_grad_norm_` → `optimizer.step()` → `lr_scheduler.step()`**
5. Increment `global_step`

Both adapters are updated in a single optimizer step. Since only the active
adapter participates in the forward graph, gradients are naturally isolated.

#### `_train_step_no_optim()`

Simplified version of `_train_step` that performs forward + backward without
the optimizer step or gradient accumulation logic. Used by the multi-adapter
training loop.

#### `_split_rollout_by_adapter()`

Partitions a rollout data dict into per-adapter subsets by reading the
`adapter_names` list and indexing all list-valued fields.

#### `update_weights()`

```python
if self._is_lora and self._num_lora_adapters > 1:
    self._sync_multi_lora_adapters(rollout_engines)
else:
    # Original merge+push path
```

In multi-LoRA mode, we **do not** merge LoRA into the base model. Instead, we
save each adapter to a temp directory and call SGLang's `load_lora_adapter`
endpoint.

#### `_sync_multi_lora_adapters()`

For each adapter:

1. `model.set_adapter(adapter_name)`
2. `save_lora_checkpoint(model, tmp_dir, adapter_name=adapter_name)`
3. `ray.get(engine.load_lora_adapter.remote(adapter_name, str(tmp_dir)))`

This replaces the previous approach of merging LoRA weights into the base model
and pushing the full model (which is O(base_model_size) per sync).

#### `save_model()`

Delegates to `checkpoint.save_multi_lora()` which saves each adapter to its own
subdirectory:

```
ckpt/iter_0000001/
└── model/
    ├── lora_a/
    │   ├── adapter_weights.pt
    │   └── adapter_config.json
    └── lora_b/
        ├── adapter_weights.pt
        └── adapter_config.json
```

#### New helper methods

- `set_active_adapter(name)` — switch adapter + restore requires_grad
- `get_lora_adapter_names()` — returns the list of adapter names

### 4. `slime/slime/backends/fsdp_utils/checkpoint.py`

**Purpose:** Multi-adapter checkpoint saving.

Added `save_multi_lora(actor, iteration, adapter_names)`:

- Creates `model/{adapter_name}/` subdirectories
- Saves each adapter separately via `save_lora_checkpoint(..., adapter_name=name)`
- Writes metadata including the list of adapter names

### 5. `slime/slime/backends/sglang_utils/sglang_engine.py`

**Purpose:** SGLang multi-LoRA mode.

| Change | Details |
|--------|---------|
| `load_lora_adapter(name, path)` | New method. POSTs to SGLang's `/load_lora_adapter` endpoint. |
| `unload_lora_adapter(name)` | New method. POSTs to SGLang's `/unload_lora_adapter` endpoint. |
| `_compute_server_args()` | When `num_lora_adapters > 1`, adds `max_loras=N` to SGLang server kwargs to enable the LoRA manager. |

### 6. `slime/slime/utils/types.py` — `Sample`

**Purpose:** Tag samples with their source adapter.

Added field:
```python
adapter_name: str | None = None
```

### 7. `slime/slime/ray/rollout.py` — `RolloutManager`

**Purpose:** Pass adapter tags through the data pipeline.

| Change | Details |
|--------|---------|
| `_convert_samples_to_train_data()` | If any sample has `adapter_name` set, adds `"adapter_names"` list to the train data dict. |
| `_split_train_data_by_dp()` | Added `"adapter_names"` to the list of keys that get partitioned by data-parallel rank, so each actor rank receives adapter tags for its local data subset. |

### 8. `openclaw-opd/openclaw_opd_api_server.py`

**Purpose:** Adapter-aware request routing.

| Change | Details |
|--------|---------|
| `__init__(args, output_queue, submission_enabled, adapter_name=None)` | New optional parameter. |
| `_handle_request()` | Sets `"model": self.adapter_name` in the forwarded request body, so SGLang routes to the correct LoRA adapter. Takes precedence over `self.served_model_name`. |
| Sample tagging | Each submitted sample gets `sample.adapter_name = self.adapter_name`. |
| Ready banner | Shows adapter name when set. |

### 9. `openclaw-opd/openclaw_opd_rollout.py` (rewritten)

**Purpose:** Dual API server instances with separate queues.

| Component | Details |
|-----------|---------|
| `_parse_adapter_ports()` | Reads `PORT_A`, `PORT_B`, `ADAPTER_NAME_A`, `ADAPTER_NAME_B` from environment. Falls back to single `PORT` for non-federated mode. |
| `AsyncRolloutWorker` | Creates 1 or 2 `OpenClawOPDAPIServer` instances, each with its own output queue. The `_is_federated` flag controls behavior. |
| `_drain_federated_queues()` | Drains both per-adapter queues until each has `rollout_batch_size` samples. All samples are tagged with adapter names. |
| `generate_rollout_openclaw_opd()` | In federated mode, calls `_drain_federated_queues` and combines results. Non-federated mode is unchanged. |

**Backward compatibility:** When `PORT_A`/`PORT_B` are not set, the rollout
worker behaves identically to the original single-server version.

### 10. `slime/train_async_federated.py` (new file)

**Purpose:** Entry point for federated training.

Thin wrapper around `train_async.py` that asserts `num_lora_adapters > 1`.
All multi-adapter logic lives in the actor and rollout worker — the training
loop itself is identical.

### 11. `openclaw-opd/run_qwen3_4b_openclaw_opd_topk_dual_lora.sh` (rewritten)

**Purpose:** Single 4-GPU Ray job with dual LoRA.

Key differences from the single-LoRA script:

| Aspect | Single-LoRA | Dual-LoRA |
|--------|-------------|-----------|
| Script | `train_async.py` | `train_async_federated.py` |
| LoRA args | `--use-lora` | `--use-lora --num-lora-adapters 2 --lora-adapter-names lora_a,lora_b` |
| Runtime env | `PORT` | `PORT_A`, `PORT_B`, `ADAPTER_NAME_A`, `ADAPTER_NAME_B` |
| GPU count | 4 | 4 (same) |

### 12. `openclaw-opd/README-federated-opd.md` (rewritten)

Updated to reflect the 4-GPU always-parallel architecture with diagrams,
training step pseudocode, and updated configuration reference.

---

## Data Flow

```
Client A → :30000 → API Server A (adapter_name="lora_a")
                       ↓ model="lora_a"
                    SGLang (multi-LoRA, routes by model name)
                       ↓
                    API Server A: sample.adapter_name = "lora_a"
                       ↓
                    Queue A
                       ↓
                    Rollout worker drains both queues
                       ↓
                    train_data["adapter_names"] = ["lora_a", ..., "lora_b", ...]
                       ↓
                    Actor._train_core_multi_adapter()
                       ↓
                    set_adapter("lora_a") → fwd/bwd
                    set_adapter("lora_b") → fwd/bwd
                    optimizer.step()
                       ↓
                    _sync_multi_lora_adapters()
                       ↓
                    SGLang reloads adapter weights
```

---

## PEFT Multi-Adapter + FSDP Interaction

### Why `set_all_lora_requires_grad` is needed

PEFT's `set_adapter("lora_a")` internally does:

```python
for module in self.model.modules():
    if isinstance(module, LoraLayer):
        # Disable inactive adapters
        module.lora_A["lora_b"].requires_grad_(False)
        module.lora_B["lora_b"].requires_grad_(False)
        # Enable active adapter
        module.lora_A["lora_a"].requires_grad_(True)
        module.lora_B["lora_a"].requires_grad_(True)
```

If we call `set_adapter("lora_a")` → forward → backward → `set_adapter("lora_b")`
→ forward → backward → `optimizer.step()`, the `lora_a` params would have
`requires_grad=False` at optimizer step time. While PyTorch's AdamW **does**
update params that have `.grad` regardless of `requires_grad`, the interaction
with FSDP2's DTensor sharding is uncertain.

To be safe, after every `set_adapter()` call we run:

```python
set_all_lora_requires_grad(model, True)
```

This ensures all LoRA params remain in the optimizer's active set throughout
the training step.

### Gradient isolation without requires_grad toggling

Even with all LoRA params having `requires_grad=True`, only the active adapter
receives gradients. This is because PEFT's `set_adapter` controls which LoRA
matrices participate in the forward computation:

- Active adapter: `output = base_layer(x) + lora_B(lora_A(x)) * scaling`
- Inactive adapter: `output = base_layer(x)` (LoRA matrices bypassed)

Since inactive params don't contribute to the forward output, they receive
zero gradient from the backward pass. The optimizer's update for these params
is `Δw = 0` (modulo weight decay, which is typically disabled for LoRA).

### Optimizer state

A single `AdamW` optimizer manages both adapter parameter sets. Adam's
momentum and variance buffers are per-parameter, so they naturally track each
adapter independently. There is no cross-contamination of optimizer state.

---

## SGLang Multi-LoRA Integration

### Server startup

When `num_lora_adapters > 1`:

```python
kwargs["max_loras"] = num_lora_adapters
```

This tells SGLang to initialize its LoRA manager, which supports dynamic
adapter loading/unloading at runtime.

### Adapter loading

After each training step, the actor saves each adapter's weights and calls:

```python
engine.load_lora_adapter.remote(adapter_name, str(adapter_dir))
```

Which translates to:

```
POST /load_lora_adapter
{"lora_name": "lora_a", "lora_path": "/tmp/slime_lora_adapters/lora_a"}
```

### Per-request routing

The API server sets `"model": self.adapter_name` in the forwarded request
body. SGLang uses this to select the correct adapter for generation.

---

## Checkpoint Resume & Adapter Merging

### Resume from a previous LoRA checkpoint

The dual-LoRA script supports `LOAD_CKPT` to resume from any LoRA checkpoint.
The checkpoint loading in `checkpoint.py` (`_load_multi_lora`) handles three
layouts:

| Checkpoint type | Detection | Behavior |
|----------------|-----------|----------|
| Multi-LoRA | `model/lora_a/adapter_weights.pt` + `model/lora_b/...` | Each adapter loads its own weights |
| Single-LoRA | `model/adapter_weights.pt` (no sub-dirs) | Weights are remapped and duplicated into all adapters |
| Merged adapter | `adapter_weights.pt` with `default` adapter name keys | Keys remapped from `.lora_A.default.` to `.lora_A.lora_a.` etc., then duplicated |

Key remapping for single→multi duplication uses regex substitution:

```python
# e.g. ".lora_A.default.weight" → ".lora_A.lora_a.weight"
new_key = re.sub(r"\.lora_([AB])\.[^.]+\.", rf".lora_\1.{target_name}.", key)
```

### `merge_lora_adapters.py`

Standalone script to weighted-average two or more LoRA adapter checkpoints.
Uses the same key normalization in reverse: strips adapter-specific names
to produce generic `default` keys, averages in float32, then saves.

Can be used in two modes:
- `--ckpt-dir` — auto-discovers sub-adapters from a dual-LoRA checkpoint
- `--adapters` — explicit paths to adapter directories

### `run_qwen3_4b_openclaw_opd_topk_dual_lora.sh` — `LOAD_CKPT`

When `LOAD_CKPT` is set, the script appends:

```bash
--load "${LOAD_CKPT}" --no-load-optim --no-load-rng --finetune
```

The optimizer starts fresh (no stale momentum from the previous run).

---

## Backward Compatibility

All changes are backward-compatible with the single-LoRA pipeline:

- `num_lora_adapters=1` (default): all new code paths are bypassed
- `PORT_A`/`PORT_B` not set: rollout worker creates a single server
- `adapter_names` not in train data: actor uses original `_train_core`
- SGLang: `max_loras` not set when `num_lora_adapters=1`
- `LOAD_CKPT` not set: no checkpoint loading (trains from scratch)

The original `run_qwen3_4b_openclaw_opd_topk_lora.sh` script and
`train_async.py` are completely unchanged and continue to work as before.
