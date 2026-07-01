#!/usr/bin/env python3
"""
Audio Synchformer/AST-token Nyström NCut visualizer.

Single-file mode:
  Input:  one MP4 with audio
  Output: one MP4 whose video stream is a static spectrogram + semi-transparent
          NCut patch-cluster mask + moving playback cursor/scroll bar, with the
          original MP4 audio track muxed back in.

Directory mode:
  Input:  a directory of MP4s
  Output: one output MP4 per input MP4. By default, AST patch tokens from every
          input video are concatenated into one global feature matrix and
          clustered with one shared NCut solve. Pass --ncut-mode per_clip to run
          a separate NCut solve for each individual input clip.

Examples:
  # Single MP4, same behavior as the earlier script.
  python audio_ast_ncut_video.py input.mp4 out.mp4 \
      --device cuda --embedder umap --n-clusters 12 --fps 30

  # Joint NCut over every MP4 in a directory.
  python audio_ast_ncut_video.py ./clips ./ncut_outputs \
      --device cuda --embedder umap --n-clusters 12 --n-eig 24

  # Joint NCut recursively, using t-SNE colors and writing a montage PNG.
  python audio_ast_ncut_video.py ./clips ./ncut_outputs \
      --recursive --embedder tsne --write-global-grid

  # Separate/per-clip NCut for every MP4 in a directory.
  python audio_ast_ncut_video.py ./clips ./ncut_outputs \
      --ncut-mode per_clip --output-suffix _single_ncut

Notes:
  * Default encoder is Synchformer's audio tower loaded from an AVCLIP
    feature-extractor checkpoint. The checkpoint contains both audio and visual
    towers; this script uses the repo's own AST wrapper to filter the audio keys.
  * Pass --encoder hf_ast to use the previous Hugging Face AST fallback.
  * Long audio is split into AST-sized log-mel windows and token patches are
    stitched back to global spectrogram coordinates.
  * Joint mode can create a very large graph. Start with a small directory and
    increase --n-eig/--n-clusters after confirming memory/time behavior.

CUDA_VISIBLE_DEVICES=0 python3 ./scripts/visualizers/ncut_audio.py \
/home/jiaray/mrBean/data/baseline_data/conductingValid_clips \
  /home/jiaray/mrBean/plots/naiive_test/audio/Validclips/perclip_10ev \
  --device cuda \
  --encoder avclip \
  --avclip-ckpt /home/jiaray/mrBean/logs/synchformer_stage1_lora_wds_ddp/26-06-25T16-58-42/checkpoints/epoch_best.pt \
  --embedder umap \
  --n-clusters 10 \
  --n-eig 10 \
  --no-feature-plot-csv


CUDA_VISIBLE_DEVICES=0 python3 ./scripts/visualizers/ncut_audio.py \
/home/jiaray/mrBean/data/baseline_data/orchestra_clips\
  /home/jiaray/mrBean/plots/baseline/audio/orchestraclips/global_perclipoverlapping50ev\
  --device cuda \
  --encoder avclip \
  --avclip-ckpt /home/jiaray/mrBean/Synchformer/checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
  --embedder umap \
  --n-clusters 50 \
  --n-eig 50 \
  --chunk-hop-frames 33 \
  --no-feature-plot-csv 



"""

from __future__ import annotations

import argparse
import shlex
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ncut_modules.audio_ncut import (
    embed_cluster_colors,
    run_ncut,
    write_feature_embedding_plot,
    _resolve_feature_plot_csv_path,
)
from ncut_modules.audio_spectrogram import write_global_grid_image
from ncut_modules.audio_tokens import (
    VideoItem,
    assign_global_offsets,
    load_model_from_args,
    prepare_video_item,
)
from ncut_modules.audio_video_output import VIDEO_EXTS, render_item
from ncut_modules.audio_metadata import (
    cluster_ids_slug,
    find_matching_parent_cache,
    load_token_cache,
    parse_cluster_id_list,
    resolve_metadata_root,
    resolve_run_metadata_dir,
    select_cluster_token_indices,
    token_cache_path,
    validate_token_cache_against_items,
    write_ncut_metadata_bundle,
)


def discover_input_videos(input_path: Path, pattern: str, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Input file does not look like a supported video: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if recursive:
        paths = sorted(p for p in input_path.rglob(pattern) if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    else:
        paths = sorted(p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not paths:
        raise FileNotFoundError(f"No videos matching {pattern!r} found in {input_path}")
    return paths


def _unique_input_names(input_roots: Sequence[Path]) -> dict[Path, str]:
    """Return stable, filesystem-safe namespace names for multiple input roots.

    When multiple input directories are passed, outputs are written under
    output/<input_dir_name>/... so clips with the same filename from different
    datasets do not overwrite each other. If two roots share the same basename,
    append a numeric suffix.
    """
    counts: dict[str, int] = {}
    names: dict[Path, str] = {}
    for root in input_roots:
        base = root.name or root.parent.name or "input"
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        names[root] = base if idx == 0 else f"{base}_{idx + 1}"
    return names


def discover_input_videos_multi(
    input_paths: Sequence[Path],
    pattern: str,
    recursive: bool,
) -> Tuple[List[Path], dict[Path, Path], dict[Path, str]]:
    """Discover videos from one or more input files/directories.

    Returns videos, a video->input_root map, and input_root->namespace map. The
    namespace map is only used when there are multiple inputs and output is a
    directory.
    """
    resolved_inputs = [p.expanduser().resolve() for p in input_paths]
    roots = [p.parent if p.is_file() else p for p in resolved_inputs]
    root_names = _unique_input_names(roots)
    videos: List[Path] = []
    video_roots: dict[Path, Path] = {}

    for input_path, root in zip(resolved_inputs, roots):
        found = discover_input_videos(input_path, pattern, recursive)
        for video in found:
            if video in video_roots:
                # Avoid doing duplicate work if the same file is reachable through
                # repeated/overlapping inputs. Keep the first root assignment.
                continue
            videos.append(video)
            video_roots[video] = root

    return videos, video_roots, root_names


def output_path_for_video(
    video_path: Path,
    input_root: Path,
    output_base: Path,
    suffix: str,
    single_file_mode: bool,
    root_namespace: str | None = None,
) -> Path:
    if single_file_mode:
        return output_base
    try:
        rel = video_path.relative_to(input_root)
    except ValueError:
        rel = Path(video_path.name)
    rel_parent = rel.parent if str(rel.parent) != "." else Path("")
    base_dir = output_base / root_namespace if root_namespace else output_base
    return base_dir / rel_parent / f"{video_path.stem}{suffix}.mp4"




NCUT_EIG_OPTION_NAMES = ("--n-eig", "--n_eig", "--num-eig", "--num_eig")
NCUT_CLUSTER_OPTION_NAMES = ("--n-clusters", "--n_clusters", "--num-clusters", "--num_clusters")
NCUT_MODE_OPTION_NAMES = ("--ncut-mode", "--ncut_mode")
NCUT_MODE_ALIASES = {
    "global": "global",
    "joint": "global",
    "per_clip": "per_clip",
    "per-clip": "per_clip",
    "single": "per_clip",
    "single_clip": "per_clip",
}


def _manual_int_option_from_argv(names: Sequence[str]) -> int | None:
    """Return the last explicit integer value for any CLI option name.

    This intentionally reads sys.argv directly so --n-eig/--n-clusters cannot be
    lost because of argparse positional parsing, shell ordering, or aliases. It
    supports both "--flag 30" and "--flag=30" forms.
    """
    value: int | None = None
    argv = sys.argv[1:]
    name_set = set(names)
    prefixes = tuple(n + "=" for n in names)
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in name_set:
            if i + 1 >= len(argv):
                raise SystemExit(f"{tok} requires an integer value")
            raw = argv[i + 1]
            try:
                value = int(raw)
            except ValueError as exc:
                raise SystemExit(f"{tok} requires an integer value, got {raw!r}") from exc
            i += 2
            continue
        for prefix in prefixes:
            if tok.startswith(prefix):
                raw = tok[len(prefix):]
                try:
                    value = int(raw)
                except ValueError as exc:
                    raise SystemExit(f"{prefix[:-1]} requires an integer value, got {raw!r}") from exc
                break
        i += 1
    return value


def _manual_str_option_from_argv(names: Sequence[str]) -> str | None:
    """Return the last explicit string value for any CLI option name."""
    value: str | None = None
    argv = sys.argv[1:]
    name_set = set(names)
    prefixes = tuple(n + "=" for n in names)
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in name_set:
            if i + 1 >= len(argv):
                raise SystemExit(f"{tok} requires a value")
            value = argv[i + 1]
            i += 2
            continue
        for prefix in prefixes:
            if tok.startswith(prefix):
                value = tok[len(prefix):]
                break
        i += 1
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "NCut on Synchformer/AST audio patch tokens. Accepts one or more input MP4s/directories "
            "of MP4s. Positional mode: the last positional path is the output. Explicit mode: use "
            "--inputs ... --output .... Default NCut mode is one separate solve per input clip."
        )
    )
    p.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help=(
            "Positional mode: input MP4 file(s) and/or directorie(s), followed by the output path. "
            "Example: in_dir_a in_dir_b out_dir, or clip.mp4 out.mp4."
        ),
    )
    p.add_argument(
        "--inputs", "--input", dest="input_flags", type=Path, nargs="+", default=None,
        help="Explicit input MP4 file(s) and/or directories. Use with --output to avoid positional ambiguity."
    )
    p.add_argument(
        "--output", dest="output_flag", type=Path, default=None,
        help="Explicit output MP4 path for one input file, or output directory for directories/multiple inputs."
    )
    p.add_argument("--glob", default="*.mp4", help="Directory-mode glob pattern. Default: *.mp4")
    p.add_argument("--recursive", action="store_true", help="Directory-mode recursive search.")
    p.add_argument("--output-suffix", default=None, help="Directory-mode suffix for each rendered MP4. Default: _single_ncut for per-clip mode, _joint_ncut for global mode.")
    p.add_argument(
        *NCUT_MODE_OPTION_NAMES,
        choices=["per_clip", "per-clip", "single", "single_clip", "global", "joint"],
        default="per_clip",
        help=(
            "NCut behavior. 'per_clip' / 'per-clip' / 'single' runs one separate NCut solve "
            "for each input clip, including a single MP4 input. 'global' / 'joint' concatenates "
            "all clips and runs one shared NCut solve. Default: per_clip."
        ),
    )
    p.add_argument("--limit-videos", type=int, default=None, help="Optional debug limit on number of input videos.")
    p.add_argument("--write-global-grid", action="store_true", help="Write joint_grid_overlay.png montage in directory mode.")
    p.add_argument("--global-grid-name", default="joint_grid_overlay.png")

    p.add_argument("--encoder", choices=["avclip", "hf_ast"], default="avclip",
                   help="Audio feature source. Default: Synchformer AVCLIP audio tower. Use hf_ast for the older Hugging Face AST fallback.")
    p.add_argument("--repo-root", type=Path, default=None,
                   help="Path to the Synchformer repo root. Usually auto-detected when running from the repo.")
    p.add_argument("--avclip-ckpt", type=Path, default=None,
                   help="Synchformer segment-level AVCLIP feature-extractor .pt checkpoint. If omitted, tries to auto-find one under logs/avclip_models.")
    p.add_argument("--max-spec-t", type=int, default=66,
                   help="Spectrogram frames per Synchformer audio segment. AVCLIP checkpoints were trained with 66 by default.")
    p.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593",
                   help="HF AST model name/path, only used with --encoder hf_ast.")
    p.add_argument("--synchformer-ckpt", type=Path, default=None,
                   help="Legacy HF-AST weight remapping path, only used with --encoder hf_ast. Prefer --avclip-ckpt.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--half", action="store_true", help="Run AST in float16. Use only on CUDA/compatible GPUs.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--num-mel-bins", type=int, default=128)
    p.add_argument("--frame-shift-ms", type=float, default=10.0)
    p.add_argument("--ast-mean", type=float, default=-4.2677393)
    p.add_argument("--ast-std", type=float, default=4.5689974)
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--chunk-hop-frames", type=int, default=None,
                   help="Hop between AST windows in mel frames. Default: AST max_length, no overlap.")
    p.add_argument(*NCUT_EIG_OPTION_NAMES, dest="n_eig", type=int, default=24,
                   help="Number of NCut eigenvectors to compute. Also accepts --num-eig/--num_eig aliases.")
    p.add_argument(*NCUT_CLUSTER_OPTION_NAMES, dest="n_clusters", type=int, default=12,
                   help="Number of final clusters. Also accepts --num-clusters/--num_clusters aliases.")
    p.add_argument("--ncut-clusterer", choices=["kmeans", "kway"], default="kmeans",
                   help="How to discretize NCut eigenvectors. Default: kmeans, which explicitly honors --n-clusters. Use kway for ncut_pytorch's native k-way discretizer.")
    p.add_argument("--embedder", choices=["umap", "tsne", "mspace"], default="umap")
    p.add_argument("--mspace-encoder-steps", type=int, default=1000,
                   help="M-space encoder training steps when --embedder mspace is used. Default: 1000.")
    p.add_argument("--mspace-decoder-steps", type=int, default=0,
                   help="M-space decoder training steps when --embedder mspace is used. Default: 0.")
    p.add_argument("--mspace-batch-size", type=int, default=1000,
                   help="M-space training/prediction batch size. Default: 1000.")
    p.add_argument("--mspace-latent-dim", type=int, default=256,
                   help="M-space hidden MLP width / latent_dim. Default: 256.")
    p.add_argument("--mspace-n-layers", type=int, default=4,
                   help="M-space MLP layer count / n_layer. Default: 4.")
    p.add_argument("--mspace-progress-bar", action="store_true",
                   help="Show ncut_pytorch M-space training progress bars.")
    p.add_argument("--no-feature-plot", action="store_true",
                   help="Disable writing the global 2D feature scatter plot.")
    p.add_argument("--feature-plot-name", default=None,
                   help="Output PNG name/path for the global feature scatter plot. Default: joint_feature_<embedder>.png in the output directory.")
    p.add_argument("--feature-plot-source", choices=["features", "ncut"], default="features",
                   help="Points to embed in the scatter plot. 'features' plots AVCLIP/AST token features; 'ncut' plots NCut eigenvectors. Default: features.")
    p.add_argument("--feature-plot-max-points", type=int, default=50000,
                   help="Max token points to draw in the feature plot. 0 means plot all points. Default: 50000.")
    p.add_argument("--feature-plot-metric", default=None,
                   help="Distance metric for the 2D plot embedding. Default: cosine for raw features, euclidean for NCut eigenvectors.")
    p.add_argument("--feature-plot-dpi", type=int, default=180)
    p.add_argument("--no-feature-plot-video-markers", action="store_true",
                   help="Do not encode source video by marker shape in the feature plot. Cluster colors are still used.")
    p.add_argument("--no-feature-plot-video-labels", action="store_true",
                   help="Do not draw source-video centroid labels on the feature plot.")
    p.add_argument("--feature-plot-video-legend-limit", type=int, default=20,
                   help="Draw a video marker legend only when the number of videos is <= this value. Default: 20.")
    p.add_argument("--feature-plot-csv-name", default=None,
                   help="CSV name/path for per-point plot metadata. Default: <feature_plot_stem>_points.csv next to the PNG.")
    p.add_argument("--no-feature-plot-csv", action="store_true",
                   help="Disable writing the CSV that maps every plotted point to video/token coordinates.")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask opacity over spectrogram.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--margin", type=int, default=48)
    p.add_argument("--scrollbar-height", type=int, default=22)
    p.add_argument("--show-title", action="store_true", help="Draw input filename at the top of each rendered video.")
    p.add_argument("--seed", type=int, default=0)

    # Token/cluster metadata. In global mode this is written by default so a
    # later cluster-specific run can reuse exact parent cluster assignments.
    p.add_argument("--no-token-metadata", action="store_true",
                   help="Disable writing token-level metadata/cache files in global mode.")
    p.add_argument("--token-metadata-dir", type=Path, default=None,
                   help="Directory for token metadata/cache files. Default: output directory, or output MP4 parent in single-file mode.")
    p.add_argument("--token-metadata-prefix", default="stage1",
                   help="Filename prefix for the normal/global NCut metadata files. Default: stage1.")
    p.add_argument("--flat-token-metadata", action="store_true",
                   help="Use the metadata root directly instead of a run-specific metadata subfolder.")
    p.add_argument("--no-token-feature-cache", action="store_true",
                   help="Do not store the raw token feature matrix in the metadata NPZ. Smaller, but less reusable.")
    p.add_argument("--save-ncut-eigvecs", action="store_true",
                   help="Also store NCut eigenvectors in the metadata NPZ.")

    # Cluster-specific / recursive NCut. This loads parent labels from a previous
    # metadata cache, selects only those parent-cluster token rows, NCuts that
    # subset jointly, then maps the new labels back onto the original spectrograms.
    p.add_argument("--parent-clusters", "--recursive-parent-clusters", "--cluster-specific-clusters",
                   dest="parent_clusters", nargs="+", default=None,
                   help="Parent cluster ids to recluster. Accepts '3,7,12' or '3, 7, 12'. Enables cluster-specific recursive NCut.")
    p.add_argument("--metadata-cache", "--parent-metadata-cache", dest="metadata_cache", type=Path, default=None,
                   help="Path to a previous *_token_cache.npz. Default: <metadata-dir>/<token-metadata-prefix>_token_cache.npz.")
    p.add_argument("--cluster-specific-prefix", default=None,
                   help="Filename prefix for recursive NCut metadata. Default: <token-metadata-prefix>_clusters_<ids>_recursive.")
    p.add_argument("--allow-metadata-mismatch", action="store_true",
                   help="Allow recursive NCut to continue if cached token metadata does not exactly match current token extraction.")
    p.add_argument("--parent-n-eig", type=int, default=None,
                   help="When auto-finding parent metadata, require this parent --n-eig value.")
    p.add_argument("--parent-n-clusters", type=int, default=None,
                   help="When auto-finding parent metadata, require this parent --n-clusters value.")
    p.add_argument("--parent-seed", type=int, default=None,
                   help="When auto-finding parent metadata, require this parent --seed value.")

    args = p.parse_args()

    # Direct argv overrides for the two parameters that must never be swallowed
    # by path parsing. This makes the final parsed values match the command line
    # even if users put options before, between, or after positional paths.
    argv_n_eig = _manual_int_option_from_argv(NCUT_EIG_OPTION_NAMES)
    argv_n_clusters = _manual_int_option_from_argv(NCUT_CLUSTER_OPTION_NAMES)
    if argv_n_eig is not None:
        args.n_eig = argv_n_eig
    if argv_n_clusters is not None:
        args.n_clusters = argv_n_clusters
    argv_clusterer = _manual_str_option_from_argv(("--ncut-clusterer",))
    if argv_clusterer is not None:
        if argv_clusterer not in {"kmeans", "kway"}:
            p.error(f"--ncut-clusterer must be one of kmeans,kway; got {argv_clusterer!r}")
        args.ncut_clusterer = argv_clusterer
    argv_ncut_mode = _manual_str_option_from_argv(NCUT_MODE_OPTION_NAMES)
    if argv_ncut_mode is not None:
        if argv_ncut_mode not in NCUT_MODE_ALIASES:
            choices = ",".join(NCUT_MODE_ALIASES)
            p.error(f"--ncut-mode must be one of {choices}; got {argv_ncut_mode!r}")
        args.ncut_mode = argv_ncut_mode

    explicit_mode = args.input_flags is not None or args.output_flag is not None
    if explicit_mode:
        if args.paths:
            p.error("do not mix positional paths with --inputs/--output; use one style or the other")
        if args.input_flags is None or args.output_flag is None:
            p.error("explicit mode requires both --inputs ... and --output ...")
        args.input = list(args.input_flags)
        args.output = args.output_flag
    else:
        if len(args.paths) < 2:
            p.error("provide at least one input path and one output path; the last positional path is the output")
        args.input = args.paths[:-1]
        args.output = args.paths[-1]

    delattr(args, "paths")
    delattr(args, "input_flags")
    delattr(args, "output_flag")

    # Normalize aliases so the rest of the script has exactly two cases.
    args.ncut_mode = NCUT_MODE_ALIASES[args.ncut_mode]

    if args.output_suffix is None:
        if args.parent_clusters:
            args.output_suffix = "_cluster_ncut"
        else:
            args.output_suffix = "_joint_ncut" if args.ncut_mode == "global" else "_single_ncut"

    if args.parent_clusters and args.ncut_mode != "global":
        p.error("--parent-clusters / recursive cluster-specific NCut requires --ncut-mode global")

    if args.n_eig < 1 or args.n_clusters < 1:
        p.error(f"--n-eig and --n-clusters must be positive; got {args.n_eig}, {args.n_clusters}")
    if args.mspace_encoder_steps < 1:
        p.error("--mspace-encoder-steps must be positive")
    if args.mspace_decoder_steps < 0:
        p.error("--mspace-decoder-steps must be >= 0")
    if args.mspace_batch_size < 1:
        p.error("--mspace-batch-size must be positive")
    if args.mspace_latent_dim < 1:
        p.error("--mspace-latent-dim must be positive")
    if args.mspace_n_layers < 0:
        p.error("--mspace-n-layers must be >= 0")

    return args


def build_mspace_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Return keyword args for ncut_pytorch.color.mspace.mspace_viz_transform."""
    return {
        "encoder_training_steps": int(args.mspace_encoder_steps),
        "decoder_training_steps": int(args.mspace_decoder_steps),
        "batch_size": int(args.mspace_batch_size),
        "latent_dim": int(args.mspace_latent_dim),
        "n_layer": int(args.mspace_n_layers),
        "progress_bar": bool(args.mspace_progress_bar),
    }


def main() -> None:
    args = parse_args()
    if args.half and not str(args.device).startswith("cuda"):
        print("--half requested on a non-CUDA device; disabling half precision.")
        args.half = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print(f"RUNNING SCRIPT: {Path(__file__).resolve()}")
    print(f"Command line argv: {shlex.join(sys.argv)}")
    print(f"NCut mode: {args.ncut_mode}")
    print(
        f"FINAL PARSED NCUT PARAMS: n_eig={args.n_eig}, "
        f"n_clusters={args.n_clusters}, clusterer={args.ncut_clusterer}"
    )
    print("=" * 80)

    mspace_kwargs = build_mspace_kwargs(args)

    input_paths = [p.expanduser().resolve() for p in args.input]
    output_path = args.output.expanduser().resolve()
    single_file_mode = len(input_paths) == 1 and input_paths[0].is_file()
    multi_input_mode = len(input_paths) > 1

    videos, video_roots, root_names = discover_input_videos_multi(input_paths, args.glob, args.recursive)
    if args.limit_videos is not None:
        videos = videos[:args.limit_videos]
    if not videos:
        raise RuntimeError("No videos selected.")

    if single_file_mode:
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("Single-file mode expects output to be an .mp4 path.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if output_path.suffix.lower() in VIDEO_EXTS:
            raise ValueError("Multi-file/directory mode expects output to be a directory, not an MP4 file path.")
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(videos)} video(s) from {len(input_paths)} input path(s).")
    if multi_input_mode:
        print("Multi-input mode: outputs will be grouped by input path name to avoid filename collisions.")
    if args.ncut_mode == "global" and len(videos) > 1:
        print("Joint/global NCut mode: all audio patch tokens from all videos will be concatenated before NCut.")
    elif args.ncut_mode == "per_clip":
        print("Per-clip NCut mode: each video will get its own independent NCut solve, including single-MP4 input.")

    model = load_model_from_args(args)

    with tempfile.TemporaryDirectory(prefix="audio_ast_ncut_") as td:
        tmp = Path(td)
        items: List[VideoItem] = []

        for i, video_path in enumerate(videos):
            input_root = video_roots[video_path]
            root_namespace = root_names[input_root] if multi_input_mode else None
            out_mp4 = output_path_for_video(
                video_path=video_path,
                input_root=input_root,
                output_base=output_path,
                suffix=args.output_suffix,
                single_file_mode=single_file_mode,
                root_namespace=root_namespace,
            )
            wav_path = tmp / f"audio_{i:05d}_16k_mono.wav"
            items.append(prepare_video_item(video_path, out_mp4, wav_path, model, args))

        overlays_for_grid: List[Tuple[Path, np.ndarray]] = []

        if args.ncut_mode == "global":
            print("\nBuilding one global token feature matrix...")
            global_features = assign_global_offsets(items)
            total_nodes = int(global_features.shape[0])
            print(f"Global token nodes: {total_nodes:,}; feature dim: {global_features.shape[1]}")
            if total_nodes > 200_000:
                print(
                    "WARNING: This is a large joint NCut problem. If it runs out of memory, "
                    "try --ncut-mode per_clip, fewer/shorter videos, a larger --chunk-hop-frames, "
                    "or a smaller --n-eig."
                )

            parent_cluster_ids = parse_cluster_id_list(args.parent_clusters)
            metadata_root = resolve_metadata_root(output_path, single_file_mode, args.token_metadata_dir)
            metadata_dir = resolve_run_metadata_dir(
                metadata_root=metadata_root,
                input_paths=input_paths,
                videos=videos,
                args=args,
                parent_cluster_ids=parent_cluster_ids,
                flat=args.flat_token_metadata,
            )
            print(f"Token metadata root: {metadata_root}")
            print(f"This run metadata dir: {metadata_dir}")

            if parent_cluster_ids:
                parent_cache_path = (
                    args.metadata_cache.expanduser().resolve()
                    if args.metadata_cache is not None
                    else find_matching_parent_cache(
                        metadata_root=metadata_root,
                        prefix=args.token_metadata_prefix,
                        items=items,
                        current_metadata_dir=metadata_dir,
                        parent_n_eig=args.parent_n_eig,
                        parent_n_clusters=args.parent_n_clusters,
                        parent_seed=args.parent_seed,
                    )
                )
                print(
                    "\nCluster-specific recursive NCut mode: loading parent cluster metadata "
                    f"from {parent_cache_path}"
                )
                parent_cache = load_token_cache(parent_cache_path)
                validate_token_cache_against_items(
                    parent_cache,
                    items,
                    allow_mismatch=args.allow_metadata_mismatch,
                )
                parent_clusters = np.asarray(parent_cache["clusters"], dtype=np.int32)
                selected_indices = select_cluster_token_indices(parent_clusters, parent_cluster_ids)
                selected_index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
                selected_features = global_features.index_select(0, selected_index_tensor)
                print(
                    f"Selected {len(selected_indices):,}/{total_nodes:,} token nodes from "
                    f"parent cluster(s) {parent_cluster_ids}."
                )

                print("Running cluster-specific Nyström NCut on selected parent-cluster tokens...")
                eigvecs, selected_clusters = run_ncut(
                    selected_features,
                    args.n_eig,
                    args.n_clusters,
                    args.device,
                    seed=args.seed,
                    clusterer=args.ncut_clusterer,
                    name=f"recursive NCut parent clusters {parent_cluster_ids}",
                )

                print(f"Coloring recursive clusters with {args.embedder.upper()}...")
                cluster_rgb = embed_cluster_colors(eigvecs, selected_clusters, args.embedder, args.seed, mspace_kwargs=mspace_kwargs)

                clusters = np.full(total_nodes, -1, dtype=np.int32)
                clusters[selected_indices] = selected_clusters.astype(np.int32)

                metadata_prefix = args.cluster_specific_prefix or (
                    f"{args.token_metadata_prefix}_clusters_{cluster_ids_slug(parent_cluster_ids)}_recursive"
                )
                if not args.no_token_metadata:
                    write_ncut_metadata_bundle(
                        metadata_dir=metadata_dir,
                        prefix=metadata_prefix,
                        stage_name="recursive_cluster_specific_ncut",
                        items=items,
                        global_features=global_features,
                        clusters=clusters,
                        cluster_rgb=cluster_rgb,
                        frame_shift_ms=args.frame_shift_ms,
                        args=args,
                        save_features=not args.no_token_feature_cache,
                        save_eigvecs=args.save_ncut_eigvecs,
                        eigvecs=eigvecs,
                        parent_clusters=parent_clusters,
                        parent_cache=parent_cache_path,
                        selected_parent_cluster_ids=parent_cluster_ids,
                        selected_global_indices=selected_indices,
                    )
                else:
                    print("Token metadata disabled for recursive NCut (--no-token-metadata).")

                if not args.no_feature_plot:
                    if args.feature_plot_source == "features":
                        plot_points = F.normalize(selected_features, dim=-1).float().cpu().numpy()
                        plot_source_name = (
                            "selected AVCLIP audio token features"
                            if args.encoder == "avclip"
                            else "selected AST audio token features"
                        )
                        plot_metric = args.feature_plot_metric or "cosine"
                    else:
                        plot_points = eigvecs.astype(np.float32)
                        plot_source_name = "recursive NCut eigenvectors"
                        plot_metric = args.feature_plot_metric or "euclidean"

                    if args.feature_plot_name is None:
                        feature_plot_out = (
                            output_path.parent / f"recursive_feature_{args.embedder}_clusters_{cluster_ids_slug(parent_cluster_ids)}.png"
                            if single_file_mode
                            else output_path / f"recursive_feature_{args.embedder}_clusters_{cluster_ids_slug(parent_cluster_ids)}.png"
                        )
                    else:
                        feature_plot_out = Path(args.feature_plot_name).expanduser()
                        if not feature_plot_out.is_absolute():
                            base_dir = output_path.parent if single_file_mode else output_path
                            feature_plot_out = base_dir / feature_plot_out

                    if args.no_feature_plot_csv:
                        feature_plot_csv_out = None
                    else:
                        csv_base_dir = output_path.parent if single_file_mode else output_path
                        feature_plot_csv_out = _resolve_feature_plot_csv_path(
                            out_png=feature_plot_out,
                            csv_name=args.feature_plot_csv_name,
                            output_base=csv_base_dir,
                        )

                    write_feature_embedding_plot(
                        points=plot_points,
                        clusters=selected_clusters,
                        cluster_rgb=cluster_rgb,
                        out_png=feature_plot_out,
                        method=args.embedder,
                        seed=args.seed,
                        max_points=args.feature_plot_max_points,
                        source_name=plot_source_name,
                        metric=plot_metric,
                        items=items,
                        frame_shift_ms=args.frame_shift_ms,
                        dpi=args.feature_plot_dpi,
                        video_markers=not args.no_feature_plot_video_markers,
                        video_labels=not args.no_feature_plot_video_labels,
                        video_legend_limit=args.feature_plot_video_legend_limit,
                        out_csv=feature_plot_csv_out,
                        write_csv=not args.no_feature_plot_csv,
                        mspace_kwargs=mspace_kwargs,
                        global_indices=selected_indices,
                    )

            else:
                print("Running one global Nyström NCut...")
                eigvecs, clusters = run_ncut(
                    global_features,
                    args.n_eig,
                    args.n_clusters,
                    args.device,
                    seed=args.seed,
                    clusterer=args.ncut_clusterer,
                    name="global NCut",
                )

                print(f"Coloring global clusters with {args.embedder.upper()}...")
                cluster_rgb = embed_cluster_colors(eigvecs, clusters, args.embedder, args.seed, mspace_kwargs=mspace_kwargs)

                if not args.no_token_metadata:
                    write_ncut_metadata_bundle(
                        metadata_dir=metadata_dir,
                        prefix=args.token_metadata_prefix,
                        stage_name="stage1_global_ncut",
                        items=items,
                        global_features=global_features,
                        clusters=clusters,
                        cluster_rgb=cluster_rgb,
                        frame_shift_ms=args.frame_shift_ms,
                        args=args,
                        save_features=not args.no_token_feature_cache,
                        save_eigvecs=args.save_ncut_eigvecs,
                        eigvecs=eigvecs,
                    )
                else:
                    print("Token metadata disabled (--no-token-metadata).")

                if not args.no_feature_plot:
                    if args.feature_plot_source == "features":
                        plot_points = F.normalize(global_features, dim=-1).float().cpu().numpy()
                        plot_source_name = "AVCLIP audio token features" if args.encoder == "avclip" else "AST audio token features"
                        plot_metric = args.feature_plot_metric or "cosine"
                    else:
                        plot_points = eigvecs.astype(np.float32)
                        plot_source_name = "NCut eigenvectors"
                        plot_metric = args.feature_plot_metric or "euclidean"

                    if args.feature_plot_name is None:
                        feature_plot_out = (
                            output_path.parent / f"joint_feature_{args.embedder}.png"
                            if single_file_mode
                            else output_path / f"joint_feature_{args.embedder}.png"
                        )
                    else:
                        feature_plot_out = Path(args.feature_plot_name).expanduser()
                        if not feature_plot_out.is_absolute():
                            base_dir = output_path.parent if single_file_mode else output_path
                            feature_plot_out = base_dir / feature_plot_out

                    if args.no_feature_plot_csv:
                        feature_plot_csv_out = None
                    else:
                        csv_base_dir = output_path.parent if single_file_mode else output_path
                        feature_plot_csv_out = _resolve_feature_plot_csv_path(
                            out_png=feature_plot_out,
                            csv_name=args.feature_plot_csv_name,
                            output_base=csv_base_dir,
                        )

                    write_feature_embedding_plot(
                        points=plot_points,
                        clusters=clusters,
                        cluster_rgb=cluster_rgb,
                        out_png=feature_plot_out,
                        method=args.embedder,
                        seed=args.seed,
                        max_points=args.feature_plot_max_points,
                        source_name=plot_source_name,
                        metric=plot_metric,
                        items=items,
                        frame_shift_ms=args.frame_shift_ms,
                        dpi=args.feature_plot_dpi,
                        video_markers=not args.no_feature_plot_video_markers,
                        video_labels=not args.no_feature_plot_video_labels,
                        video_legend_limit=args.feature_plot_video_legend_limit,
                        out_csv=feature_plot_csv_out,
                        write_csv=not args.no_feature_plot_csv,
                        mspace_kwargs=mspace_kwargs,
                    )

            render_source = "recursive cluster-specific NCut labels" if parent_cluster_ids else "global NCut labels"
            for i, item in enumerate(items):
                print(f"\n=== Rendering {item.input_mp4.name} from {render_source} ===")
                item_clusters = clusters[item.token_start:item.token_end]
                if len(item_clusters) != item.token_count:
                    raise RuntimeError(f"Internal split error for {item.input_mp4}")
                temp_video = tmp / f"video_{i:05d}_no_audio.mp4"
                overlay = render_item(item, item_clusters, cluster_rgb, temp_video, args)
                if args.write_global_grid:
                    overlays_for_grid.append((item.input_mp4, overlay))

        else:
            if args.write_global_grid:
                print("\nPer-clip mode: the grid montage will show independently colored per-clip NCut overlays.")
            if not args.no_feature_plot:
                print("\nPer-clip mode: skipping the global feature scatter plot. Use --ncut-mode global for a shared plot.")

            for i, item in enumerate(items):
                print(f"\n=== Running per-clip Nyström NCut for {item.input_mp4.name} ===")
                item_features = item.grid.features
                total_nodes = int(item_features.shape[0])
                print(f"{item.input_mp4.name}: {total_nodes:,} token nodes; feature dim: {item_features.shape[1]}")
                if total_nodes > 200_000:
                    print(
                        "WARNING: This is a large per-clip NCut problem. If it runs out of memory, "
                        "try a larger --chunk-hop-frames or a smaller --n-eig."
                    )

                eigvecs, item_clusters = run_ncut(item_features, args.n_eig, args.n_clusters, args.device, seed=args.seed + i, clusterer=args.ncut_clusterer, name=f"per-clip NCut {item.input_mp4.name}")
                print(f"Coloring clusters for {item.input_mp4.name} with {args.embedder.upper()}...")
                cluster_rgb = embed_cluster_colors(eigvecs, item_clusters, args.embedder, args.seed, mspace_kwargs=mspace_kwargs)

                temp_video = tmp / f"video_{i:05d}_no_audio.mp4"
                overlay = render_item(item, item_clusters, cluster_rgb, temp_video, args)
                if args.write_global_grid:
                    overlays_for_grid.append((item.input_mp4, overlay))

        if args.write_global_grid:
            grid_out = output_path.parent / args.global_grid_name if single_file_mode else output_path / args.global_grid_name
            write_global_grid_image(overlays_for_grid, grid_out)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
