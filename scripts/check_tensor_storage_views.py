#!/usr/bin/env python
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = "/home/jiaray/mrBean/Synchformer"
sys.path.insert(0, REPO)

from scripts.train_utils import get_transforms, get_datasets


def env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v.lower() in {"1", "true", "yes", "y", "on"}


def logical_bytes(x):
    if not torch.is_tensor(x):
        return 0
    return x.numel() * x.element_size()


def storage_bytes(x):
    if not torch.is_tensor(x):
        return 0
    try:
        return x.untyped_storage().nbytes()
    except Exception:
        return x.storage().size() * x.element_size()


def tensor_report(name, x):
    if not torch.is_tensor(x):
        print(f"{name}: not tensor: {type(x)}")
        return

    lb = logical_bytes(x)
    sb = storage_bytes(x)
    ratio = sb / max(lb, 1)

    print(
        f"{name}: "
        f"shape={tuple(x.shape)} "
        f"dtype={x.dtype} "
        f"contiguous={x.is_contiguous()} "
        f"storage_offset={x.storage_offset()} "
        f"logical_mb={lb / 1024 / 1024:.2f} "
        f"storage_mb={sb / 1024 / 1024:.2f} "
        f"storage/logical={ratio:.2f} "
        f"data_ptr={x.data_ptr()} "
        f"storage_ptr={x.untyped_storage().data_ptr()}",
        flush=True,
    )

    if ratio > 1.5:
        print(f"!!! {name} probably retains larger backing storage", flush=True)
    else:
        print(f"OK: {name} storage roughly matches logical tensor size", flush=True)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    torch.set_num_threads(1)

    cfg = OmegaConf.load(f"{REPO}/configs/sync.yaml")

    cfg.data.dataset.target = "dataset.webdataset_tar_inmemory_cached_sync.WebDatasetTarInMemoryCachedSync"

    data_dir = os.environ.get(
        "DATA_DIR",
        "/home/jiaray/mrBean/data/webdataset_clips/smoke_train",
    )

    cfg.data.vids_path = data_dir
    cfg.data.dataset.params.train_vids_dir = data_dir
    cfg.data.dataset.params.valid_vids_dir = data_dir
    cfg.data.dataset.params.test_vids_dir = data_dir

    cfg.data.dataset.params.cache_decoded = env_bool("CACHE_DECODED", False)
    cfg.data.dataset.params.decoded_cache_size = int(os.environ.get("DECODED_CACHE_SIZE", "0"))
    cfg.data.dataset.params.cache_tar_handles = env_bool("CACHE_TAR_HANDLES", False)
    cfg.data.dataset.params.tar_handle_cache_size = int(os.environ.get("TAR_HANDLE_CACHE_SIZE", "8"))

    cfg.data.dataset.params.debug_io = False
    cfg.data.dataset.params.debug_signal = False
    cfg.data.dataset.params.worker_threads = 1
    cfg.data.dataset.params.decode_threads = 1
    cfg.data.dataset.params.strict_video_fps = 25
    cfg.data.dataset.params.strict_audio_fps = 16000
    cfg.data.dataset.params.max_clip_len_sec = None

    transforms = get_transforms(cfg)
    datasets = get_datasets(cfg, transforms, which_datasets=["train"])

    ds = datasets["train"]
    print(f"dataset length: {len(ds)}")
    print(f"DATA_DIR: {data_dir}")

    indices_env = os.environ.get("INDICES", "0,1,10,11,50,100,200,300,400")
    indices = [int(x) for x in indices_env.split(",") if x.strip()]

    for idx in indices:
        if idx >= len(ds):
            continue

        print("\n" + "=" * 100)
        print(f"INDEX {idx}")

        item = ds[idx]

        print(f"path: {item.get('path')}")
        tensor_report("item['video']", item["video"])
        tensor_report("item['audio']", item["audio"])

        if "targets" in item:
            for k, v in item["targets"].items():
                if torch.is_tensor(v):
                    tensor_report(f"item['targets']['{k}']", v)

        # Force cleanup between samples so this test isolates returned storage,
        # not Python references from previous loop iterations.
        del item


if __name__ == "__main__":
    main()
