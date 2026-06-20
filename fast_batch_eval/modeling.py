from __future__ import annotations

from typing import List, Tuple

import torch
from omegaconf import OmegaConf

from dataset.transforms import make_class_grid
from scripts.train_utils import get_model, get_transforms
from utils.utils import check_if_file_exists_else_download

# Reuse the repo's config patching logic from example.py.
from example import patch_config

from .utils import safe_prefix


DEFAULT_MODEL_SPECS = [
    # ("lrs3", "23-12-23T18-33-57"),
    ("vggsound", "24-01-02T10-00-53"),
    ("audioset", "24-01-04T16-39-21"),
]


def parse_model_specs(args) -> List[Tuple[str, str]]:
    """
    Default: use the published pretrained sync models listed above.

    Optional override:
        --exp_names EXP1 EXP2 ...
        --model_names NAME1 NAME2 ...
    """
    if args.exp_names is None:
        return DEFAULT_MODEL_SPECS

    if args.model_names is None:
        model_names = [safe_prefix(exp_name) for exp_name in args.exp_names]
    else:
        model_names = args.model_names

    if len(model_names) != len(args.exp_names):
        raise RuntimeError("--model_names must have the same number of entries as --exp_names")

    return list(zip(model_names, args.exp_names))


def load_synchformer(
    exp_name: str,
    device: torch.device,
    *,
    torch_compile: bool = False,
    torch_compile_mode: str = "reduce-overhead",
):
    """Load one pretrained Synchformer sync model and its test transform."""
    cfg_path = f"./logs/sync_models/{exp_name}/cfg-{exp_name}.yaml"
    ckpt_path = f"./logs/sync_models/{exp_name}/{exp_name}.pt"

    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    cfg = OmegaConf.load(cfg_path)
    cfg = patch_config(cfg)

    _, model = get_model(cfg, device)

    ckpt = torch.load(
        ckpt_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    if torch_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--torch_compile requested, but this PyTorch build has no torch.compile")
        model = torch.compile(model, mode=torch_compile_mode)

    transform = get_transforms(cfg, ["test"])["test"]

    max_off_sec = cfg.data.max_off_sec
    num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
    grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)

    return cfg, model, transform, grid
