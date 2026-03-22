"""Merge (weighted average) two or more LoRA adapter checkpoints into one.

Takes adapter checkpoint directories (each containing ``adapter_weights.pt``
and optionally ``adapter_config.json``), computes a weighted average of the
LoRA weight tensors, and saves the result as a new adapter checkpoint.

Usage:
    # Equal average (default)
    python merge_lora_adapters.py \
        --adapters /path/to/lora_a /path/to/lora_b \
        --output /path/to/merged_adapter

    # Weighted: 70% lora_a + 30% lora_b
    python merge_lora_adapters.py \
        --adapters /path/to/lora_a /path/to/lora_b \
        --weights 0.7 0.3 \
        --output /path/to/merged_adapter

    # From a dual-LoRA training checkpoint (auto-detect sub-adapters)
    python merge_lora_adapters.py \
        --ckpt-dir /path/to/ckpt/qwen3-4b-dual-lora \
        --output /path/to/merged_adapter

The merged adapter can be loaded as a regular single-LoRA checkpoint with
the standard training scripts (via ``--load``).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def find_latest_model_dir(ckpt_dir: Path) -> Path:
    tracker = ckpt_dir / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        step = int(tracker.read_text().strip())
        model_dir = ckpt_dir / f"iter_{step:07d}" / "model"
        if model_dir.exists():
            return model_dir
    iter_dirs = sorted(ckpt_dir.glob("iter_*/model"), key=lambda p: p.parent.name)
    if iter_dirs:
        return iter_dirs[-1]
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def discover_adapters_in_model_dir(model_dir: Path) -> list[Path]:
    """Find adapter sub-directories inside a multi-LoRA model checkpoint."""
    candidates = []
    for child in sorted(model_dir.iterdir()):
        if child.is_dir() and (child / "adapter_weights.pt").exists():
            candidates.append(child)
    return candidates


def load_adapter_weights(adapter_dir: Path) -> dict[str, torch.Tensor]:
    weights_file = adapter_dir / "adapter_weights.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"No adapter_weights.pt in {adapter_dir}")
    state = torch.load(weights_file, map_location="cpu", weights_only=True)
    print(f"  Loaded {len(state)} tensors from {adapter_dir}")
    return state


def normalize_key(key: str) -> str:
    """Strip adapter-specific name segments to produce a generic LoRA key.

    Multi-adapter checkpoints save keys like:
        base_model.model.layers.0.self_attn.q_proj.lora_A.lora_a.weight
    We normalize to:
        base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    so that the merged output can be loaded as a single "default" adapter.
    """
    import re
    key = re.sub(r"\.lora_A\.[^.]+\.", ".lora_A.default.", key)
    key = re.sub(r"\.lora_B\.[^.]+\.", ".lora_B.default.", key)
    return key


def merge_adapters(
    adapter_states: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Weighted average of adapter state dicts."""
    assert len(adapter_states) == len(weights)

    normalized_states = []
    for state in adapter_states:
        normalized_states.append({normalize_key(k): v for k, v in state.items()})

    all_keys = set()
    for state in normalized_states:
        all_keys.update(state.keys())

    merged = {}
    for key in sorted(all_keys):
        tensors = []
        w_list = []
        for state, w in zip(normalized_states, weights):
            if key in state:
                tensors.append(state[key].float())
                w_list.append(w)
        if not tensors:
            continue
        w_sum = sum(w_list)
        avg = sum(t * (w / w_sum) for t, w in zip(tensors, w_list))
        merged[key] = avg.to(tensors[0].dtype)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Weighted average of LoRA adapter checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapters", nargs="+", type=str, default=None,
        help="Paths to adapter directories (each with adapter_weights.pt)",
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=None,
        help="Path to a dual-LoRA training checkpoint dir (auto-discovers sub-adapters from latest iteration)",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Per-adapter weights (default: equal). Must match number of adapters.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for the merged adapter",
    )

    args = parser.parse_args()

    # Resolve adapter paths
    adapter_dirs: list[Path] = []
    if args.ckpt_dir:
        model_dir = find_latest_model_dir(Path(args.ckpt_dir))
        adapter_dirs = discover_adapters_in_model_dir(model_dir)
        if not adapter_dirs:
            if (model_dir / "adapter_weights.pt").exists():
                print(f"Single adapter found at {model_dir}; nothing to merge.")
                sys.exit(0)
            raise FileNotFoundError(f"No adapter sub-directories in {model_dir}")
        print(f"Auto-discovered {len(adapter_dirs)} adapters in {model_dir}:")
        for d in adapter_dirs:
            print(f"  - {d.name}")
    elif args.adapters:
        adapter_dirs = [Path(p) for p in args.adapters]
    else:
        parser.error("Provide --adapters or --ckpt-dir")

    if len(adapter_dirs) < 2:
        print("Only one adapter found; copying as-is (no averaging needed).")
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(adapter_dirs[0], out, dirs_exist_ok=True)
        print(f"Copied to {out}")
        return

    # Resolve weights
    weights = args.weights
    if weights is None:
        weights = [1.0] * len(adapter_dirs)
    if len(weights) != len(adapter_dirs):
        parser.error(f"--weights has {len(weights)} values but found {len(adapter_dirs)} adapters")

    w_sum = sum(weights)
    print(f"\nMerging {len(adapter_dirs)} adapters with weights:")
    for d, w in zip(adapter_dirs, weights):
        print(f"  {d.name:20s}  weight={w:.4f}  ({w/w_sum*100:.1f}%)")

    # Load all adapter states
    adapter_states = [load_adapter_weights(d) for d in adapter_dirs]

    # Merge
    merged = merge_adapters(adapter_states, weights)
    print(f"\nMerged result: {len(merged)} tensors")

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out / "adapter_weights.pt")

    # Copy adapter_config.json from the first adapter (they share the same config)
    for d in adapter_dirs:
        cfg = d / "adapter_config.json"
        if cfg.exists():
            shutil.copy2(cfg, out / "adapter_config.json")
            break

    print(f"\nSaved merged adapter to {out}")
    print("This can be used as a single-LoRA checkpoint with --load, or as")
    print("the starting point for dual-LoRA training with LORA_INIT_CKPT.")


if __name__ == "__main__":
    main()
