"""LoRA utilities for FSDP-based training.

Provides helpers to apply HuggingFace PEFT LoRA adapters to a model,
save/load LoRA-only checkpoints, and merge/unmerge adapters for weight sync.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _build_lora_config(args: Namespace):
    """Build a ``LoraConfig`` from CLI arguments."""
    from peft import LoraConfig

    target_modules = None
    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    modules_to_save = None
    if args.lora_modules_to_save:
        modules_to_save = [m.strip() for m in args.lora_modules_to_save.split(",")]

    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
        task_type="CAUSAL_LM",
    )


def apply_lora(
    model: torch.nn.Module,
    args: Namespace,
    adapter_name: str = "default",
) -> torch.nn.Module:
    """Wrap *model* with PEFT LoRA adapters according to *args*.

    Returns the PeftModel wrapper.  All base-model parameters are frozen;
    only the LoRA parameters are trainable.
    """
    from peft import get_peft_model

    lora_config = _build_lora_config(args)
    model = get_peft_model(model, lora_config, adapter_name=adapter_name)

    if dist.get_rank() == 0:
        model.print_trainable_parameters()

    return model


def add_lora_adapter(
    model: torch.nn.Module,
    args: Namespace,
    adapter_name: str = "lora_b",
) -> None:
    """Add an additional named LoRA adapter to an existing PeftModel."""
    lora_config = _build_lora_config(args)
    model.add_adapter(adapter_name, lora_config)
    if dist.get_rank() == 0:
        logger.info("Added LoRA adapter %r", adapter_name)


def set_all_lora_requires_grad(model: torch.nn.Module, requires_grad: bool = True) -> None:
    """Force ``requires_grad`` on every LoRA parameter regardless of the active adapter.

    PEFT's ``set_adapter`` toggles requires_grad per-adapter.  When training
    multiple adapters in the same optimizer, we keep them all trainable and
    rely on the forward graph to route gradients only to the active adapter.
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(requires_grad)


def propagate_no_split_modules(model: torch.nn.Module) -> torch.nn.Module:
    """Ensure ``_no_split_modules`` is visible through the PEFT wrapper.

    PEFT wraps the original model as ``model.base_model.model``.  FSDP's
    ``apply_fsdp2`` reads ``model._no_split_modules`` to decide which layers
    to shard individually.  This helper copies the attribute up when missing.
    """
    if getattr(model, "_no_split_modules", None):
        return model

    # PeftModel -> LoraModel -> original HF model
    inner = getattr(model, "base_model", None)
    if inner is not None:
        inner = getattr(inner, "model", inner)
    if inner is not None:
        no_split = getattr(inner, "_no_split_modules", None)
        if no_split:
            model._no_split_modules = no_split
            logger.info(f"Propagated _no_split_modules from inner model: {no_split}")

    return model


def save_lora_checkpoint(
    model: torch.nn.Module,
    path: Path,
    adapter_name: str | None = None,
) -> None:
    """Save only the LoRA adapter weights + config to *path*.

    When *adapter_name* is given, only keys belonging to that adapter are saved.
    Only rank 0 writes to disk.  All ranks participate in the state-dict
    gathering (handled by FSDP through ``state_dict()``).
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    full_state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    if dist.get_rank() == 0:
        lora_state = {k: v for k, v in full_state.items() if "lora_" in k}
        if adapter_name is not None:
            lora_state = {
                k: v for k, v in lora_state.items()
                if f".{adapter_name}." in k or k.endswith(f".{adapter_name}")
            }
        path.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state, path / "adapter_weights.pt")

        if hasattr(model, "peft_config"):
            import json

            target_cfg_name = adapter_name or next(iter(model.peft_config), None)
            if target_cfg_name and target_cfg_name in model.peft_config:
                cfg_dict = model.peft_config[target_cfg_name].to_dict()
                for k, v in cfg_dict.items():
                    if isinstance(v, set):
                        cfg_dict[k] = sorted(v)
                with open(path / "adapter_config.json", "w") as f:
                    json.dump(cfg_dict, f, indent=2)

        logger.info(f"Saved LoRA adapter {adapter_name!r} ({len(lora_state)} tensors) to {path}")

    dist.barrier()


def load_lora_checkpoint(
    model: torch.nn.Module,
    path: Path,
    adapter_name: str | None = None,
) -> None:
    """Load LoRA adapter weights from *path* into *model*.

    When *adapter_name* is given, only parameters matching that adapter name
    are loaded.  Broadcasts from rank 0 to all other ranks.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    adapter_file = path / "adapter_weights.pt"
    if not adapter_file.exists():
        logger.warning(f"No LoRA adapter found at {adapter_file}; skipping load.")
        return

    if dist.get_rank() == 0:
        lora_state = torch.load(adapter_file, map_location="cpu", weights_only=True)
        logger.info(f"Loaded LoRA adapter ({len(lora_state)} tensors) from {path}")
    else:
        lora_state = {}

    full_state = {}
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        if adapter_name is not None and f".{adapter_name}." not in name and not name.endswith(f".{adapter_name}"):
            continue
        if name in lora_state:
            full_state[name] = lora_state[name]

    if full_state:
        set_model_state_dict(
            model,
            full_state,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=True,
                strict=False,
            ),
        )
        logger.info(f"Loaded {len(full_state)} LoRA parameters from checkpoint.")

    dist.barrier()


def get_merged_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a merged (base + LoRA) state dict on CPU.

    Merges the adapter in-place, copies the state dict, then unmerges.
    The returned dict uses base-model keys (no ``lora_`` or ``base_model.`` prefixes).
    """
    model.merge_adapter()
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        merged_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        # Strip PEFT wrapper prefixes so keys match the original HF model
        cleaned = {}
        for k, v in merged_state.items():
            # Skip lora-specific keys (they've been merged into base)
            if "lora_" in k:
                continue
            # Strip common PEFT prefixes
            clean_key = k
            for prefix in ["base_model.model.", "base_model."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            cleaned[clean_key] = v
        return cleaned
    finally:
        model.unmerge_adapter()
