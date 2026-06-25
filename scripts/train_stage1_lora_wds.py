#!/usr/bin/env python3
"""
Standalone Synchformer stage-1 LoRA fine-tuning runner for MP4-in-tar WebDataset shards.

This script DOES NOT patch Synchformer repo files and does NOT monkey-patch imported functions.
It imports Synchformer modules normally, builds the AVCLIP model, loads an existing stage-1
checkpoint, wraps selected Linear layers with LoRA modules in this process only, and trains.

It also saves two checkpoint forms:
  1) checkpoints/epoch_latest.pt and epoch_best.pt
     Synchformer-stage2-compatible merged checkpoint. The LoRA delta is merged into the
     original Linear weights, so the state_dict keys look like the original AVCLIP model.
  2) checkpoints/lora_latest.pt
     LoRA-training resume checkpoint. Use --resume-lora with this file to continue LoRA training.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf, open_dict


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear: y = base(x) + scale * B(A(dropout(x)))."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)

        # Standard LoRA init: nonzero A, zero B, so the wrapped model initially equals the base model.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        # Compute in fp32 for numerical stability, then cast back to the base dtype.
        delta = self.lora_B.weight.float().matmul(self.lora_A.weight.float()) * self.scaling
        return (self.base.weight.float() + delta).to(dtype=self.base.weight.dtype)

    @torch.no_grad()
    def merged_bias(self) -> Optional[torch.Tensor]:
        if self.base.bias is None:
            return None
        return self.base.bias.detach().clone()


def add_repo_to_path(repo: Path) -> None:
    repo = repo.resolve()
    train_clip_src = repo / "model" / "modules" / "feat_extractors" / "train_clip_src"
    for p in (str(repo), str(train_clip_src)):
        if p not in sys.path:
            sys.path.insert(0, p)


def clean_state_dict_keys(sd: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = dict(sd)
    # Common DDP prefixes.
    for prefix in ("module.", "model."):
        if out and all(k.startswith(prefix) for k in out.keys()):
            out = {k[len(prefix):]: v for k, v in out.items()}
    return out

def ensure_train_clip_compat_defaults(cfg: Any) -> None:
    """
    Fill missing OpenCLIP/Synchformer training keys expected by train_one_epoch().
    This only modifies the in-memory cfg object in this standalone script.
    It does not patch repo files or config files.
    """
    with open_dict(cfg):
        if "distill" not in cfg:
            cfg.distill = False

        if "transform_sequence_train" not in cfg:
            if "data" in cfg and "transform_sequence_train" in cfg.data:
                cfg.transform_sequence_train = cfg.data.transform_sequence_train
            else:
                cfg.transform_sequence_train = []

        if "transform_sequence_test" not in cfg:
            if "data" in cfg and "transform_sequence_test" in cfg.data:
                cfg.transform_sequence_test = cfg.data.transform_sequence_test
            else:
                cfg.transform_sequence_test = []

        if "training" in cfg and "skip_scheduler" not in cfg.training:
            cfg.training.skip_scheduler = False

def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], Mapping):
                return clean_state_dict_keys(ckpt[key])
        # Bare state dict: values should mostly be tensors.
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return clean_state_dict_keys(ckpt)
    raise RuntimeError(
        "Could not find a model state_dict in checkpoint. Expected one of: "
        "checkpoint['state_dict'], checkpoint['model'], checkpoint['model_state_dict'], or a bare state dict."
    )


def load_initial_stage1_checkpoint(
    model: nn.Module,
    ckpt_path: Path,
    *,
    strict: bool = True,
) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = extract_state_dict(ckpt)
    status = model.load_state_dict(sd, strict=strict)
    if not strict:
        missing = list(getattr(status, "missing_keys", []))
        unexpected = list(getattr(status, "unexpected_keys", []))
        if missing or unexpected:
            logging.warning("Non-strict S1 load: missing=%s unexpected=%s", missing[:50], unexpected[:50])
    logging.info("Loaded initial stage-1 checkpoint: %s", ckpt_path)


def get_parent_module(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def target_regex_from_mode(mode: str, custom_regex: Optional[str]) -> re.Pattern[str]:
    if custom_regex:
        return re.compile(custom_regex)

    if mode == "attention":
        # AST HF-style examples: attention.attention.query/key/value, attention.output.dense
        # MotionFormer/timm-style examples: attn.qkv, attn.proj
        pat = r"(^|\.)(attn|attention)(\.|$).*(qkv|query|key|value|proj|out_proj|dense)$|(^|\.)(qkv|query|key|value|out_proj)$"
    elif mode == "attention_mlp":
        pat = r"(^|\.)(attn|attention|mlp|intermediate|output)(\.|$).*(qkv|query|key|value|proj|out_proj|dense|fc1|fc2)$|(^|\.)(qkv|query|key|value|out_proj|fc1|fc2)$"
    elif mode == "all_linear":
        pat = r".*"
    else:
        raise ValueError(f"Unknown --lora-target-mode={mode!r}")
    return re.compile(pat)


def apply_lora_to_module(
    root: nn.Module,
    *,
    name_prefix: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_re: re.Pattern[str],
    exclude_re: Optional[re.Pattern[str]],
) -> List[str]:
    replaced: List[str] = []

    # list(...) because we mutate the module tree.
    for local_name, module in list(root.named_modules()):
        if local_name == "" or not isinstance(module, nn.Linear):
            continue
        full_name = f"{name_prefix}.{local_name}" if name_prefix else local_name
        if not target_re.search(local_name) and not target_re.search(full_name):
            continue
        if exclude_re is not None and (exclude_re.search(local_name) or exclude_re.search(full_name)):
            continue

        parent, child_name = get_parent_module(root, local_name)

        # Do not wrap torch.nn.MultiheadAttention.out_proj.
        # MultiheadAttention.forward() directly reads out_proj.weight/out_proj.bias
        # instead of calling out_proj(x), so replacing it with LoRALinear breaks forward().
        # Also, LoRA would not actually be applied there unless we replaced the whole MHA.
        if isinstance(parent, nn.MultiheadAttention):
            logging.info("Skipping LoRA for nn.MultiheadAttention child: %s", full_name)
            continue

        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(full_name)
    return replaced


def set_trainable_policy(
    model: nn.Module,
    *,
    train_logit_scale: bool,
    train_layer_norm: bool,
    train_bias: bool,
    train_proj: bool,
) -> None:
    # Freeze everything first. LoRA modules are inserted after this in the caller, and their params
    # remain trainable. Then this function is called again to unfreeze explicitly allowed extras.
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        lname = name.lower()
        keep = "lora_" in lname
        if train_logit_scale and "logit_scale" in lname:
            keep = True
        if train_layer_norm and ("norm" in lname or "ln" in lname or "layernorm" in lname):
            keep = True
        if train_bias and lname.endswith(".bias"):
            keep = True
        if train_proj and ("aproj" in lname or "vproj" in lname):
            keep = True
        p.requires_grad = keep


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def merged_original_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict with LoRA deltas merged and original Linear key names restored.

    If a module named 'a.b.c' is LoRALinear, this emits:
      a.b.c.weight = base.weight + LoRA_delta
      a.b.c.bias   = base.bias, if present
    and skips all a.b.c.base.*, a.b.c.lora_A.*, a.b.c.lora_B.* keys.
    """
    model = unwrap_model(model)
    lora_modules = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    lora_prefixes = tuple(name + "." for name in lora_modules.keys())

    merged: Dict[str, torch.Tensor] = {}
    raw = model.state_dict()

    for key, value in raw.items():
        if any(key.startswith(prefix) for prefix in lora_prefixes):
            continue
        merged[key] = value.detach().cpu().clone()

    for name, module in lora_modules.items():
        merged[f"{name}.weight"] = module.merged_weight().detach().cpu().clone()
        bias = module.merged_bias()
        if bias is not None:
            merged[f"{name}.bias"] = bias.detach().cpu().clone()

    return merged


def lora_resume_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()}


def trainable_parameter_report(model: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    return trainable, total, pct


def build_data_loaders(cfg: Any, transforms: Mapping[str, Any], DataInfoCls: Any) -> Dict[str, Any]:
    """
    Build Synchformer stage-1 sparsesync-style loaders but forward *all* data.dataset.params.
    This is intentionally local to this script; it does not modify training.data.get_data().
    """
    from utils.utils import get_obj_from_str

    DatasetClass = get_obj_from_str(cfg.data.dataset.target)
    params = OmegaConf.to_container(cfg.data.dataset.params, resolve=True)
    params = {} if params is None else dict(params)

    def make_dataset(split: str, transform_key: str) -> torch.utils.data.Dataset:
        phase_params = dict(params)
        if "size_ratios" in phase_params and phase_params["size_ratios"] is not None:
            phase_params["size_ratio"] = phase_params["size_ratios"].get(split, phase_params.get("size_ratio", None))
        phase_params.pop("size_ratios", None)
        return DatasetClass(
            split=split,
            vids_dir=cfg.data.vids_path,
            transforms=transforms[transform_key],
            **phase_params,
        )

    data: Dict[str, Any] = {}
    for split, transform_key, is_train in (("train", "train", True), ("valid", "test", False)):
        dataset = make_dataset(split, transform_key)
        sampler = DistributedSampler(dataset, shuffle=is_train) if getattr(cfg, "distributed", False) else None
        shuffle = bool(is_train and sampler is None)

        num_workers = int(cfg.training.get("num_workers", 0))
        loader_kwargs = dict(
            batch_size=int(cfg.training.base_batch_size),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=bool(cfg.training.get("pin_memory", True)),
            drop_last=True,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(cfg.training.get("persistent_workers", False))
            if "prefetch_factor" in cfg.training and cfg.training.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(cfg.training.prefetch_factor)

        dataloader = DataLoader(dataset, **loader_kwargs)
        dataloader.num_samples = len(dataset)
        dataloader.num_batches = len(dataloader)
        data[split] = DataInfoCls(dataloader=dataloader, sampler=sampler)
        logging.info(
            "Loaded %s: samples=%d batches=%d batch_size=%s workers=%s prefetch=%s persistent=%s",
            split,
            len(dataset),
            len(dataloader),
            cfg.training.base_batch_size,
            num_workers,
            cfg.training.get("prefetch_factor", None),
            cfg.training.get("persistent_workers", None),
        )
    return data


def metric_value(metrics: Optional[Mapping[str, Any]], metric_name: str) -> Optional[float]:
    if not metrics:
        return None
    if metric_name in metrics:
        try:
            return float(metrics[metric_name])
        except Exception:
            return None
    # Be forgiving: original code uses metric_name='precision'. Some loggers prefix phase names.
    candidates = [k for k in metrics.keys() if k.endswith("/" + metric_name) or k.endswith(metric_name)]
    for k in candidates:
        try:
            return float(metrics[k])
        except Exception:
            pass
    return None


def save_checkpoints(
    *,
    cfg: Any,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    metrics: Optional[Mapping[str, Any]],
    is_best: bool,
) -> None:
    ckpt_dir = Path(cfg.checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    merged_sd = merged_original_state_dict(model)
    stage2_compatible = {
        "epoch": epoch,
        "name": cfg.name,
        "state_dict": merged_sd,
        "metrics": {"sync_w_shifts": dict(metrics or {})},
        "args": cfg,
        "lora_merged": True,
        "note": "LoRA deltas merged into original AVCLIP Linear weights; intended for Synchformer stage-2 ckpt_path.",
    }

    latest_path = ckpt_dir / "epoch_latest.pt"
    tmp_path = ckpt_dir / "tmp_epoch_latest.pt"
    torch.save(stage2_compatible, tmp_path)
    os.replace(tmp_path, latest_path)
    logging.info("Saved merged stage2-compatible latest checkpoint: %s", latest_path)

    if is_best:
        best_path = ckpt_dir / "epoch_best.pt"
        tmp_best = ckpt_dir / "tmp_epoch_best.pt"
        torch.save(stage2_compatible, tmp_best)
        os.replace(tmp_best, best_path)
        logging.info("Saved merged stage2-compatible BEST checkpoint: %s", best_path)

    lora_resume = {
        "epoch": epoch,
        "name": cfg.name,
        "lora_model_state_dict": lora_resume_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "metrics": {"sync_w_shifts": dict(metrics or {})},
        "args": cfg,
        "lora_resume": True,
    }
    if scaler is not None:
        lora_resume["scaler"] = scaler.state_dict()

    lora_path = ckpt_dir / "lora_latest.pt"
    tmp_lora = ckpt_dir / "tmp_lora_latest.pt"
    torch.save(lora_resume, tmp_lora)
    os.replace(tmp_lora, lora_path)
    logging.info("Saved LoRA-resume checkpoint: %s", lora_path)


def add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: Optional[str] = None,
) -> None:
    """Backport argparse.BooleanOptionalAction for Python 3.8 environments."""
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help)
    parser.add_argument(f"--no-{name.lstrip('--')}", dest=dest, action="store_false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path, help="Path to Synchformer repo root")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/segment_avclip.yaml")
    parser.add_argument("--s1-ckpt", required=True, type=Path, help="Existing stage-1 AVCLIP checkpoint to adapt")
    parser.add_argument("--resume-lora", default=None, type=Path, help="Resume LoRA training from checkpoints/lora_latest.pt")
    parser.add_argument("--allow-nonstrict-s1-load", action="store_true", help="Allow non-strict initial S1 checkpoint load")

    parser.add_argument("--lora-scope", choices=("audio", "visual", "both"), default="both")
    parser.add_argument("--lora-target-mode", choices=("attention", "attention_mlp", "all_linear"), default="attention")
    parser.add_argument("--lora-target-regex", default=None, help="Override target regex for Linear module names")
    parser.add_argument("--lora-exclude-regex", default=r"(^|\.)(head|classifier)(\.|$)")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    add_boolean_optional_argument(parser, "--train-logit-scale", default=True)
    add_boolean_optional_argument(parser, "--train-layer-norm", default=False)
    add_boolean_optional_argument(parser, "--train-bias", default=False)
    add_boolean_optional_argument(parser, "--train-proj", default=False)

    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="OmegaConf dotlist overrides after --, e.g. -- training.num_epochs=3 logging.use_wandb=false",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    add_repo_to_path(repo)
    os.chdir(repo)

    # Imports happen after sys.path is set.
    from main import set_env_variables
    from scripts.train_utils import get_curr_time_w_random_shift, get_transforms
    from utils.utils import cfg_sanity_check_and_patch
    from model.modules.feat_extractors.train_clip_src.open_clip.factory import create_model
    from model.modules.feat_extractors.train_clip_src.training.train_clip import DistributedDataParallel
    from training.data import DataInfo
    from training.distributed import broadcast_object, init_distributed_device, is_master
    from training.file_utils import pt_load  # noqa: F401 - imported to fail early if package path is wrong
    from training.logger import setup_logging
    from training.scheduler import const_lr, const_lr_cooldown, cosine_lr
    from training.train import evaluate_on_sync_w_shifts, train_one_epoch

    set_env_variables()

    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    cfg_yml = OmegaConf.load(args.config)
    cfg_cli = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)

    if "start_time" not in cfg or cfg.start_time is None:
        cfg.start_time = get_curr_time_w_random_shift()
    try:
        OmegaConf.register_new_resolver("add", lambda *xs: sum(xs), replace=True)
    except TypeError:
        # Older OmegaConf has no replace=.
        try:
            OmegaConf.register_new_resolver("add", lambda *xs: sum(xs))
        except ValueError:
            pass
    OmegaConf.resolve(cfg)
    cfg_sanity_check_and_patch(cfg)
    ensure_train_clip_compat_defaults(cfg)

    if cfg.action != "train_avclip":
        raise RuntimeError(f"This script is only for stage-1 AVCLIP training. Got cfg.action={cfg.action!r}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(cfg)
    date_str = cfg.get("start_time", get_curr_time_w_random_shift())
    if cfg.distributed:
        date_str = broadcast_object(cfg, date_str)
    cfg.name = date_str

    log_base_path = Path(cfg.logging.logdir) / cfg.name
    log_base_path.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_path = str(log_base_path / "checkpoints")
    Path(cfg.checkpoint_path).mkdir(parents=True, exist_ok=True)
    cfg.tensorboard_path = str(log_base_path)

    log_path = log_base_path / (f"out-{cfg.rank}.log" if cfg.logging.get("log_local", False) else "out.log")
    setup_logging(str(log_path), logging.DEBUG if cfg.debug else logging.INFO)

    if bool(cfg.logging.get("use_wandb", False)):
        raise RuntimeError("This standalone LoRA runner intentionally does not initialize wandb. Set logging.use_wandb=false.")

    logging.info("Command: %s", " ".join(sys.argv))
    logging.info("Repo: %s", repo)
    logging.info("Config: %s", args.config)
    logging.info("Initial S1 checkpoint: %s", args.s1_ckpt)
    logging.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # Build base AVCLIP model and load the existing stage-1 checkpoint BEFORE inserting LoRA.
    model = create_model(cfg, device)
    load_initial_stage1_checkpoint(model, args.s1_ckpt, strict=not args.allow_nonstrict_s1_load)

    # Freeze base model, insert LoRA, then apply final trainable policy.
    for p in model.parameters():
        p.requires_grad = False

    target_re = target_regex_from_mode(args.lora_target_mode, args.lora_target_regex)
    exclude_re = re.compile(args.lora_exclude_regex) if args.lora_exclude_regex else None

    replaced: List[str] = []
    if args.lora_scope in ("audio", "both"):
        if not hasattr(model, "a_encoder"):
            raise RuntimeError("Model has no a_encoder; cannot apply audio LoRA.")
        replaced += apply_lora_to_module(
            model.a_encoder,
            name_prefix="a_encoder",
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_re=target_re,
            exclude_re=exclude_re,
        )
    if args.lora_scope in ("visual", "both"):
        if not hasattr(model, "v_encoder"):
            raise RuntimeError("Model has no v_encoder; cannot apply visual LoRA.")
        replaced += apply_lora_to_module(
            model.v_encoder,
            name_prefix="v_encoder",
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_re=target_re,
            exclude_re=exclude_re,
        )

    if not replaced:
        raise RuntimeError(
            "LoRA did not replace any Linear layers. Try --lora-target-mode attention_mlp or --lora-target-mode all_linear, "
            "or pass --lora-target-regex."
        )

    set_trainable_policy(
        model,
        train_logit_scale=args.train_logit_scale,
        train_layer_norm=args.train_layer_norm,
        train_bias=args.train_bias,
        train_proj=args.train_proj,
    )

    logging.info("LoRA replaced %d Linear layers", len(replaced))
    for name in replaced[:200]:
        logging.info("  LoRA: %s", name)
    if len(replaced) > 200:
        logging.info("  ... %d more", len(replaced) - 200)

    trainable, total, pct = trainable_parameter_report(model)
    logging.info("Trainable params after LoRA: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", pct)

    transforms = get_transforms(cfg)
    data = build_data_loaders(cfg, transforms, DataInfo)
    assert len(data), "At least one train/eval dataset is required."

    if cfg.distributed:
        ddp_args = {}
        if cfg.training.get("ddp_static_graph", False):
            ddp_args["static_graph"] = True
        model = DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # Resume LoRA training after wrapping, before optimizer-state load.
    start_epoch = 0
    resume_payload = None
    if args.resume_lora is not None:
        resume_payload = torch.load(str(args.resume_lora), map_location="cpu")
        if "lora_model_state_dict" not in resume_payload:
            raise RuntimeError(f"--resume-lora checkpoint lacks lora_model_state_dict: {args.resume_lora}")
        missing, unexpected = unwrap_model(model).load_state_dict(resume_payload["lora_model_state_dict"], strict=False)
        if missing or unexpected:
            logging.warning("LoRA resume non-strict load: missing=%s unexpected=%s", missing[:50], unexpected[:50])
        start_epoch = int(resume_payload.get("epoch", 0))
        logging.info("Resumed LoRA model from %s at epoch=%d", args.resume_lora, start_epoch)

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_param = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_param = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_param, "weight_decay": 0.0},
            {"params": rest_param, "weight_decay": float(cfg.training.optimizer.weight_decay)},
        ],
        lr=float(cfg.training.learning_rate),
        betas=tuple(cfg.training.optimizer.betas),
        eps=1e-8,
    )
    scaler = GradScaler() if cfg.training.precision == "amp" else None

    if resume_payload is not None:
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        if scaler is not None and "scaler" in resume_payload:
            scaler.load_state_dict(resume_payload["scaler"])

    scheduler = None
    if "train" in data and not bool(cfg.training.get("skip_scheduler", False)):
        total_steps = data["train"].dataloader.num_batches * int(cfg.training.num_epochs)
        warmup = int(cfg.training.lr_scheduler.get("warmup", 0) // max(1, int(cfg.world_size)))
        if cfg.training.lr_scheduler.name == "cosine":
            scheduler = cosine_lr(optimizer, float(cfg.training.learning_rate), warmup, total_steps)
        elif cfg.training.lr_scheduler.name == "const":
            scheduler = const_lr(optimizer, float(cfg.training.learning_rate), warmup, total_steps)
        elif cfg.training.lr_scheduler.name == "const-cooldown":
            cooldown_steps = data["train"].dataloader.num_batches * int(cfg.training.epochs_cooldown)
            scheduler = const_lr_cooldown(
                optimizer,
                float(cfg.training.learning_rate),
                warmup,
                total_steps,
                cooldown_steps,
                float(cfg.training.lr_cooldown_power),
                float(cfg.training.lr_cooldown_end),
            )
        else:
            raise RuntimeError(f"Unknown lr scheduler: {cfg.training.lr_scheduler.name}")

    cfg.save_logs = bool(cfg.logging.logdir and str(cfg.logging.logdir).lower() != "none" and is_master(cfg))
    if is_master(cfg):
        cfg_path = log_base_path / f"cfg-{cfg.name}.yaml"
        OmegaConf.save(cfg, cfg_path)
        logging.info("Saved resolved config: %s", cfg_path)

    best_metric: Optional[float] = None
    metric_name = str(cfg.training.get("metric_name", "precision"))
    maximize = bool(cfg.training.get("to_max_metric", True))

    writer = None
    loss = None
    dist_model = None

    for epoch in range(start_epoch, int(cfg.training.num_epochs)):
        if is_master(cfg):
            logging.info("Start epoch %d", epoch)
        if "train" in data:
            data["train"].set_epoch(epoch)
        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, cfg, writer)

        completed_epoch = epoch + 1
        metrics = None
        if bool(cfg.training.get("run_shifted_win_val", True)) and int(cfg.training.get("val_frequency", 1)) > 0:
            if (epoch % int(cfg.training.val_frequency)) == 0 or completed_epoch == int(cfg.training.num_epochs):
                metrics = evaluate_on_sync_w_shifts(model, data, "valid", completed_epoch, cfg, writer, loss)

        curr = metric_value(metrics, metric_name)
        is_best = False
        if curr is not None:
            if best_metric is None:
                is_best = True
            elif maximize:
                is_best = curr > best_metric
            else:
                is_best = curr < best_metric
            if is_best:
                best_metric = curr
                logging.info("New best %s: %.6f", metric_name, curr)
        else:
            # If validation metrics are unavailable, make the first epoch best and always save latest.
            is_best = best_metric is None
            if best_metric is None:
                best_metric = float("-inf") if maximize else float("inf")

        if is_master(cfg) and cfg.save_logs:
            save_checkpoints(
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=completed_epoch,
                metrics=metrics,
                is_best=is_best,
            )

    logging.info("Done. Checkpoints are under: %s", cfg.checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
