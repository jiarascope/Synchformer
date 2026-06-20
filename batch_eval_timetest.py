#!/usr/bin/env python3
"""
Profile tar MP4/audio decode + Synchformer forward time on one tar shard.

Run this from the Synchformer repo root, with the fast modular eval package importable.
Recommended first run uses --num_workers 0 so batch fetch time is an honest serial
measure of tar read + PyAV decode + transforms + collate, without DataLoader prefetch hiding it.


run it like:

with cpu:

python3 batch_eval_timetest.py \
  --tar /home/jiaray/mrBean/data/webdataset_clips/0REJ-lCGiKU.tar \
  --device cpu \
  --batch_size 4 \
  --num_workers 0 \
  --max_samples 32 \
  --data_iter 1 \
  --out_csv /home/jiaray/mrBean/tables/profile_cpu_1.csv



with gpu: 

CUDA_VISIBLE_DEVICES=0 python3 batch_eval_timetest.py \
  --tar /home/jiaray/mrBean/data/baseline_data/tarfiles/Rak6idcESSk.tar \
  --device cuda:0 \
  --batch_size 4 \
  --num_workers 0 \
  --max_samples 32 \
  --data_iter 1 \
  --out_csv /home/jiaray/mrBean/tables/profile_cpu_.csv

CUDA_VISIBLE_DEVICES=0 python3 batch_eval_timetest.py \
  --tar /home/jiaray/mrBean/data/baseline_data/tarfiles/Rak6idcESSk.tar \
  --device cuda:0 \
  --batch_size 8 \
  --num_workers 8 \
  --persistent_workers \
  --prefetch_factor 4 \
  --max_samples 64 \
  --data_iter 1 \
  --out_csv /home/jiaray/mrBean/tables/profile_gpu.csv


"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import shlex
import statistics as stats
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from scripts.train_utils import prepare_inputs


def _import_any(names: Iterable[str]):
    last_exc = None
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Could not import any of: {list(names)}. Last error: {last_exc}")


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
    return xs[idx]


def summarize(name: str, values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "name": name,
            "n": 0,
            "mean_s": float("nan"),
            "median_s": float("nan"),
            "p90_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
            "total_s": 0.0,
        }

    return {
        "name": name,
        "n": len(values),
        "mean_s": stats.mean(values),
        "median_s": stats.median(values),
        "p90_s": percentile(values, 0.90),
        "min_s": min(values),
        "max_s": max(values),
        "total_s": sum(values),
    }


def print_summary(rows: List[Dict[str, Any]]):
    print("\n=== Timing summary ===")
    print(f"{'stage':<24} {'n':>5} {'mean':>10} {'median':>10} {'p90':>10} {'total':>10}")
    for r in rows:
        print(
            f"{r['name']:<24} {r['n']:>5} "
            f"{r['mean_s']:>10.4f} "
            f"{r['median_s']:>10.4f} "
            f"{r['p90_s']:>10.4f} "
            f"{r['total_s']:>10.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare tar decode/data loading time against Synchformer forward time."
    )

    parser.add_argument("--tar", required=True, help="Path to one .tar/.tar.gz/.tgz shard")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Use 0 for clean serial decode-vs-model timing. With workers > 0, "
            "DataLoader prefetch overlaps CPU decode with GPU model work."
        ),
    )
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")

    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--data_iter",
        type=int,
        default=1,
        help="Use 1 to avoid decoded-cache repeats hiding decode cost.",
    )

    # Important: default None. Do not silently clip 10s videos to 5s.
    parser.add_argument(
        "--max_clip_len_sec",
        type=float,
        default=None,
        help="Optional decode cap. Leave unset for real full-clip pipeline timing.",
    )

    parser.add_argument("--strict_video_fps", type=float, default=None)
    parser.add_argument("--strict_audio_fps", type=float, default=None)

    parser.add_argument(
        "--cache_decoded",
        action="store_true",
        help="Enable decoded tensor cache. Useful for data_iter > 1.",
    )
    parser.add_argument("--decoded_cache_size", type=int, default=64)

    parser.add_argument("--no_cache_tar_handles", dest="cache_tar_handles", action="store_false")
    parser.set_defaults(cache_tar_handles=True)
    parser.add_argument("--tar_handle_cache_size", type=int, default=8)

    parser.add_argument("--clone_cached_tensors", action="store_true")

    parser.add_argument("--exp_name", default="24-01-02T10-00-53", help="Default: vggsound sync model")
    parser.add_argument("--model_name", default="vggsound")

    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--torch_compile_mode", default="reduce-overhead")

    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument("--out_csv", default=None)

    parser.add_argument(
        "--package",
        default="fast_batch_eval",
        help="Package containing datasets.py/modeling.py/utils.py.",
    )

    return parser


def make_run_meta(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "argv_json": json.dumps(sys.argv),
        "script": sys.argv[0],

        "tar": args.tar,
        "device": args.device,
        "batch_size_arg": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": bool(args.persistent_workers),
        "prefetch_factor": args.prefetch_factor,

        "max_samples": args.max_samples,
        "max_batches": args.max_batches,
        "data_iter": args.data_iter,
        "max_clip_len_sec": args.max_clip_len_sec,

        "cache_decoded": bool(args.cache_decoded),
        "decoded_cache_size": args.decoded_cache_size,
        "cache_tar_handles": bool(args.cache_tar_handles),
        "tar_handle_cache_size": args.tar_handle_cache_size,
        "clone_cached_tensors": bool(args.clone_cached_tensors),

        "strict_video_fps": args.strict_video_fps,
        "strict_audio_fps": args.strict_audio_fps,

        "model_name": args.model_name,
        "exp_name": args.exp_name,
        "torch_compile": bool(args.torch_compile),
        "torch_compile_mode": args.torch_compile_mode,
        "warmup_batches": args.warmup_batches,
        "package": args.package,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_meta = make_run_meta(args)

    device = torch.device(args.device)

    datasets_mod = _import_any([f"{args.package}.datasets", "datasets"])
    modeling_mod = _import_any([f"{args.package}.modeling", "modeling"])

    eval_args = SimpleNamespace(
        tar_dir=args.tar,
        recursive=False,

        cache_decoded=args.cache_decoded,
        decoded_cache_size=args.decoded_cache_size,

        cache_tar_handles=args.cache_tar_handles,
        tar_handle_cache_size=args.tar_handle_cache_size,

        clone_cached_tensors=args.clone_cached_tensors,

        strict_video_fps=args.strict_video_fps,
        strict_audio_fps=args.strict_audio_fps,
        max_clip_len_sec=args.max_clip_len_sec,

        max_samples=args.max_samples,
        data_iter=args.data_iter,
    )

    print(f"Command: {run_meta['command']}")
    print(f"Loading Synchformer model {args.model_name}: {args.exp_name}")

    cfg, model, transform, _grid = modeling_mod.load_synchformer(
        args.exp_name,
        device,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
    )

    # CPU cannot run Conv3D fp16.
    if device.type == "cpu":
        if hasattr(cfg, "training") and hasattr(cfg.training, "use_half_precision"):
            cfg.training.use_half_precision = False
        model = model.float()

    print("Building dataset")
    dataset, num_clips = datasets_mod.build_eval_dataset(eval_args, transform)
    print(f"Clips: {num_clips}; trials: {len(dataset)}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0 and args.persistent_workers),
    )

    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)
    it = iter(loader)

    data_wait_times: List[float] = []
    h2d_prepare_times: List[float] = []
    model_forward_times: List[float] = []
    end_to_end_times: List[float] = []
    per_batch_rows: List[Dict[str, Any]] = []

    batch_idx = 0

    print("\nProfiling batches")

    while True:
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        batch_total_t0 = time.perf_counter()

        # Time waiting for the next dataloader batch.
        # With num_workers=0, this is roughly tar read + PyAV decode + transforms + collate.
        # With num_workers>0, this is only how long the main loop waited after prefetching.
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        data_wait = t1 - t0

        # Time input preparation / transfer to device.
        synchronize(device)
        t2 = time.perf_counter()
        aud, vid, _targets = prepare_inputs(batch, device)

        if device.type == "cpu":
            aud = aud.float()
            vid = vid.float()

        synchronize(device)
        t3 = time.perf_counter()
        h2d_prepare = t3 - t2

        # Time model forward only.
        synchronize(device)
        t4 = time.perf_counter()

        with torch.inference_mode():
            if device.type == "cuda":
                use_amp = bool(getattr(cfg.training, "use_half_precision", False))
                with torch.autocast("cuda", enabled=use_amp):
                    _ = model(vid, aud)
            else:
                _ = model(vid, aud)

        synchronize(device)
        t5 = time.perf_counter()
        model_forward = t5 - t4

        batch_total = time.perf_counter() - batch_total_t0

        is_warmup = batch_idx < args.warmup_batches
        phase = "warmup" if is_warmup else "timed"

        batch_path = batch.get("path")
        if isinstance(batch_path, (list, tuple)):
            bs = len(batch_path)
            sample_ids = list(batch_path)
        else:
            bs = args.batch_size
            sample_ids = [str(batch_path)]

        print(
            f"batch={batch_idx:04d} {phase:<6} bs={bs:<3} "
            f"data_wait={data_wait:.4f}s "
            f"prepare_h2d={h2d_prepare:.4f}s "
            f"model_forward={model_forward:.4f}s "
            f"end_to_end={batch_total:.4f}s"
        )

        row = {
            **run_meta,

            "batch_idx": batch_idx,
            "phase": phase,
            "batch_size": bs,

            "data_wait_s": data_wait,
            "prepare_h2d_s": h2d_prepare,
            "model_forward_s": model_forward,
            "end_to_end_s": batch_total,

            "data_wait_per_clip_s": data_wait / bs if bs else float("nan"),
            "prepare_h2d_per_clip_s": h2d_prepare / bs if bs else float("nan"),
            "model_forward_per_clip_s": model_forward / bs if bs else float("nan"),
            "end_to_end_per_clip_s": batch_total / bs if bs else float("nan"),

            "sample_ids_json": json.dumps(sample_ids),
        }

        per_batch_rows.append(row)

        if not is_warmup:
            data_wait_times.append(data_wait)
            h2d_prepare_times.append(h2d_prepare)
            model_forward_times.append(model_forward)
            end_to_end_times.append(batch_total)

        batch_idx += 1

    summary_rows = [
        summarize("data_wait", data_wait_times),
        summarize("prepare_h2d", h2d_prepare_times),
        summarize("model_forward", model_forward_times),
        summarize("end_to_end", end_to_end_times),
    ]

    print_summary(summary_rows)

    data_total = sum(data_wait_times)
    prepare_total = sum(h2d_prepare_times)
    model_total = sum(model_forward_times)
    denom = data_total + prepare_total + model_total

    print("\n=== Relative cost over timed batches ===")
    if denom > 0:
        print(f"data_wait share:     {100.0 * data_total / denom:.1f}%")
        print(f"prepare_h2d share:   {100.0 * prepare_total / denom:.1f}%")
        print(f"model_forward share: {100.0 * model_total / denom:.1f}%")

    if model_total > 0:
        print(f"data_wait / model_forward: {data_total / model_total:.2f}x")

    if args.num_workers > 0:
        print(
            "\nNote: num_workers > 0 enables DataLoader prefetch/overlap. "
            "data_wait is how long the main training/eval loop waited for the next batch, "
            "not the full raw CPU decode time."
        )
    else:
        print(
            "\nNote: with --num_workers 0, data_wait is approximately tar read + PyAV decode "
            "+ Synchformer test transform + PyTorch collation for each batch."
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w", newline="") as f:
            fieldnames = list(per_batch_rows[0].keys()) if per_batch_rows else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if per_batch_rows:
                writer.writeheader()
                writer.writerows(per_batch_rows)

        print(f"Saved per-batch timings: {out}")


if __name__ == "__main__":
    main()