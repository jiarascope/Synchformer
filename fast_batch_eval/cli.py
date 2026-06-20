from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .fields import SUMMARY_FIELDNAMES, TRIAL_FIELDNAMES
from .modeling import parse_model_specs
from .runner import evaluate_one_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast modular stochastic Synchformer eval over tar-sharded MP4 clips."
    )

    parser.add_argument(
        "--tar_dir",
        required=True,
        help="Directory containing WebDataset .tar/.tar.gz/.tgz shards, one tar shard, glob, or brace pattern.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_iter", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out_csv", default="synchformer_stochastic_trial_predictions.csv")
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Aggregate metrics CSV. Defaults to <out_csv stem>_summary.csv.",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional debug limit on number of clips before stochastic repetition.",
    )

    # Dataset/decode controls.
    parser.add_argument(
        "--strict_video_fps",
        type=float,
        default=None,
        help="Optional check, e.g. 25. Raises if decoded FPS differs.",
    )
    parser.add_argument(
        "--strict_audio_fps",
        type=float,
        default=None,
        help="Optional check, e.g. 16000. Raises if decoded audio rate differs.",
    )
    parser.add_argument(
        "--max_clip_len_sec",
        type=float,
        default=None,
        help="Optional maximum seconds decoded per MP4 sample.",
    )
    parser.add_argument(
        "--no_cache_decoded",
        dest="cache_decoded",
        action="store_false",
        help="Disable decoded rgb/audio cache. This is slower but uses less RAM.",
    )
    parser.set_defaults(cache_decoded=True)
    parser.add_argument(
        "--decoded_cache_size",
        type=int,
        default=64,
        help="Decoded clip cache size per worker. Use 0 for unlimited per worker.",
    )
    parser.add_argument(
        "--no_cache_tar_handles",
        dest="cache_tar_handles",
        action="store_false",
        help="Disable tarfile handle cache.",
    )
    parser.set_defaults(cache_tar_handles=True)
    parser.add_argument(
        "--tar_handle_cache_size",
        type=int,
        default=8,
        help="Open tar handle cache size per worker. Use 0 for unlimited per worker.",
    )
    parser.add_argument(
        "--clone_cached_tensors",
        action="store_true",
        help=(
            "Clone cached raw rgb/audio tensors before transform. Safer if transforms mutate "
            "in place, but slower and uses more memory bandwidth."
        ),
    )

    # DataLoader/GPU controls.
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive. Strongly recommended when num_workers > 0.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Optionally wrap model in torch.compile. Can help after warmup but may fail on some models.",
    )
    parser.add_argument(
        "--torch_compile_mode",
        default="reduce-overhead",
        help="torch.compile mode if --torch_compile is set.",
    )

    # Output/behavior.
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Write error rows and keep going if a batch fails.",
    )
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument(
        "--exp_names",
        nargs="+",
        default=None,
        help="Optional list of Synchformer experiment IDs. Defaults to published sync models.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="Optional friendly names for --exp_names. Must match length of --exp_names.",
    )

    return parser


def validate_args(args):
    if args.data_iter <= 0:
        raise RuntimeError(f"--data_iter must be positive, got {args.data_iter}")
    if args.batch_size <= 0:
        raise RuntimeError(f"--batch_size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise RuntimeError(f"--num_workers must be >= 0, got {args.num_workers}")
    if args.topk <= 0:
        raise RuntimeError(f"--topk must be positive, got {args.topk}")
    if args.decoded_cache_size < 0:
        raise RuntimeError("--decoded_cache_size must be >= 0. Use 0 for unlimited.")
    if args.tar_handle_cache_size < 0:
        raise RuntimeError("--tar_handle_cache_size must be >= 0. Use 0 for unlimited.")
    if args.prefetch_factor <= 0:
        raise RuntimeError("--prefetch_factor must be positive")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_csv is None:
        summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    else:
        summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_specs = parse_model_specs(args)

    print("Models to evaluate:")
    for model_name, exp_name in model_specs:
        print(f"  {model_name}: {exp_name}")

    print("")
    print("Fast path:")
    print("  - data_iter is flattened into the dataset index order")
    print("  - repeated trials for the same clip are adjacent")
    print("  - decoded MP4 rgb/audio can be cached per DataLoader worker")
    print("  - softmax/top-k are computed on GPU before compact CPU copies")
    print(f"Per-trial predictions CSV: {out_csv}")
    print(f"Aggregate summary CSV:     {summary_csv}")

    with open(out_csv, "w", newline="") as trial_f, open(summary_csv, "w", newline="") as summary_f:
        trial_writer = csv.DictWriter(trial_f, fieldnames=TRIAL_FIELDNAMES)
        summary_writer = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDNAMES)
        trial_writer.writeheader()
        summary_writer.writeheader()

        for model_name, exp_name in model_specs:
            evaluate_one_model(
                model_name=model_name,
                exp_name=exp_name,
                args=args,
                trial_writer=trial_writer,
                summary_writer=summary_writer,
                device=device,
            )
            trial_f.flush()
            summary_f.flush()

    print("")
    print(f"Saved per-trial predictions: {out_csv}")
    print(f"Saved aggregate summary:     {summary_csv}")


if __name__ == "__main__":
    main()
