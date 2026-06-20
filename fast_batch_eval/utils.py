from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional

import torch


def safe_prefix(name: str) -> str:
    """Make a stable prefix/name for CSV output."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")


def extract_tar_name(sample_id: str) -> str:
    """Return tar shard name from IDs like /path/shard.tar::clip.mp4."""
    tar_part = str(sample_id).split("::", 1)[0]
    return Path(tar_part).name or tar_part


def offset_key(offset_sec: float) -> str:
    """Stable JSON key for offset seconds."""
    return f"{float(offset_sec):.6g}"


def coerce_paths(batch_path: Any) -> List[str]:
    """Default collate leaves a list/tuple of strings for batch['path']."""
    if isinstance(batch_path, (list, tuple)):
        return [str(p) for p in batch_path]
    return [str(batch_path)]


def tensor_to_cpu_1d(value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert collated target values to a 1D CPU tensor."""
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu()
    else:
        out = torch.as_tensor(value)

    if dtype is not None:
        out = out.to(dtype=dtype)

    if out.ndim == 0:
        out = out.unsqueeze(0)

    return out.reshape(-1)


def unpack_offset_logits(logits: Any) -> torch.Tensor:
    """
    For the plain sync model, logits is usually a tensor.
    Some repo actions return tuples/lists/dicts; take the offset logits.
    """
    if isinstance(logits, torch.Tensor):
        return logits

    if isinstance(logits, dict):
        if "offset" in logits:
            return logits["offset"]
        for value in logits.values():
            if isinstance(value, torch.Tensor):
                return value
        raise RuntimeError(f"Could not find tensor logits in dict keys: {list(logits.keys())}")

    if isinstance(logits, (tuple, list)):
        if not logits:
            raise RuntimeError("Empty logits tuple/list")
        return logits[0]

    raise RuntimeError(f"Unsupported logits type: {type(logits)}")


def make_amp_context(cfg, device: torch.device):
    use_amp = bool(getattr(cfg.training, "use_half_precision", False))
    amp_enabled = use_amp and device.type == "cuda"

    if device.type == "cuda":
        return torch.autocast("cuda", enabled=amp_enabled)
    return contextlib.nullcontext()


def get_target_class_and_offset(targets: dict[str, Any], grid_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Synchformer's TemporalCropAndOffset writes:
        targets['offset_target'] = class index for grid offsets
        targets['offset_sec'] = sampled/applied offset in seconds

    This function is intentionally strict: if these are missing, the transform
    probably did not run or the Dataset returned already-processed targets.
    """
    if "offset_target" not in targets:
        raise RuntimeError(
            "targets['offset_target'] is missing. The TemporalCropAndOffset transform "
            "probably did not run, or the Dataset returned already-processed data "
            "without target labels."
        )

    target_idx = tensor_to_cpu_1d(targets["offset_target"], dtype=torch.long)

    if "offset_sec" in targets:
        target_offset = tensor_to_cpu_1d(targets["offset_sec"], dtype=torch.float32)
    else:
        target_offset = grid_tensor[target_idx]

    return target_idx, target_offset
