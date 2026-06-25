#!/usr/bin/env python3
"""
Create Synchformer Stage-1 AVCLIP segment similarity matrices for one or more videos,
including videos that are longer than the model's fixed inference window.

Run from the root of https://github.com/v-iashin/Synchformer after activating the
synchformer environment.

Long-video behavior:
  By default, each input video is split into fixed-duration chunks, each chunk is
  passed through the Stage-1 AVCLIP model, segment features are concatenated, and
  full-video similarity matrices are computed from all extracted segment features.

/home/jiaray/mrBean/data/baseline_data/conducting_clips

Example:
  python scripts/visualizers/visualize_avclip_similarity_any_length.py \
    --cfg configs/segment_avclip.yaml \
    --checkpoint checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
    --vids /path/to/long_video.mp4 \
    --out /path/to/similarity.png \
    --npz /path/to/similarity.npz

python scripts/visualizers/visualize_avclip_similarity.py \
  --cfg configs/segment_avclip.yaml \
  --checkpoint checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
  --video-dir /home/jiaray/mrBean/data/baseline_data/conducting_clips \
  --out /home/jiaray/mrBean/plots/baseline

"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def add_synchformer_repo_to_path() -> Path:
    """Make top-level Synchformer imports work when this script is run by path."""
    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), *here.parents]
    for candidate in candidates:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir() and (candidate / "utils").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = add_synchformer_repo_to_path()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf

from dataset.dataset_utils import get_video_and_audio
from scripts.train_utils import get_model, get_transforms, prepare_inputs
from utils.utils import which_ffmpeg


DEFAULT_VFPS = 25
DEFAULT_AFPS = 16000
DEFAULT_SIZE_BEFORE_CROP = 256


def get_ffprobe() -> str:
    """Return an ffprobe executable path if available."""
    ffmpeg = which_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe")
        if sibling.exists():
            return str(sibling)
    return shutil.which("ffprobe") or ""


def get_media_duration_sec(path: str) -> float:
    """Read media duration without loading the full video into memory."""
    ffprobe = get_ffprobe()
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = float(json.loads(proc.stdout)["format"]["duration"])
        if duration > 0:
            return duration

    # Fallback only; this can be memory-heavy for very long videos.
    _, _, info = torchvision.io.read_video(path, pts_unit="sec")
    if "duration" in info:
        return float(info["duration"])
    raise RuntimeError(f"Could not determine media duration for {path}; install ffprobe or pass --whole-video.")


def make_chunk_starts(duration: float, chunk_seconds: float, stride_seconds: float) -> List[float]:
    """Return start times that cover the full video, including the tail."""
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be > 0")
    if stride_seconds <= 0:
        raise ValueError("--stride-seconds must be > 0")
    if duration <= 0:
        raise ValueError("Video duration must be > 0")
    if duration <= chunk_seconds:
        return [0.0]

    max_start = max(0.0, duration - chunk_seconds)
    starts = list(np.arange(0.0, max_start + 1e-6, stride_seconds, dtype=float))
    if not starts or abs(starts[-1] - max_start) > 1e-3:
        starts.append(max_start)

    # Round for cleaner filenames and de-duplicate after the tail append.
    deduped = sorted({round(float(s), 3) for s in starts})
    return deduped


def reencode_video(path: str, vfps: int, afps: int, in_size: int, out_dir: Path) -> str:
    """Adapted from Synchformer's example.py."""
    assert which_ffmpeg() != "", "ffmpeg was not found. Activate the Synchformer env or install ffmpeg."
    out_dir.mkdir(exist_ok=True, parents=True)
    new_path = out_dir / f"{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4"

    cmd = [
        which_ffmpeg(), "-hide_banner", "-loglevel", "panic", "-y", "-i", path,
        "-vf",
        f"fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',"
        "crop='trunc(iw/2)'*2:'trunc(ih/2)'*2",
        "-ar", str(afps),
        "-ac", "1",
        str(new_path),
    ]
    subprocess.check_call(cmd)

    wav_path = str(new_path).replace(".mp4", ".wav")
    cmd = [
        which_ffmpeg(), "-hide_banner", "-loglevel", "panic", "-y", "-i", str(new_path),
        "-acodec", "pcm_s16le", "-ac", "1", wav_path,
    ]
    subprocess.call(cmd)
    return str(new_path)


def reencode_video_chunk(
    path: str,
    start_sec: float,
    chunk_seconds: float,
    vfps: int,
    afps: int,
    in_size: int,
    out_dir: Path,
    chunk_index: int,
) -> str:
    """Create one normalized fixed-duration chunk for long-video inference."""
    assert which_ffmpeg() != "", "ffmpeg was not found. Activate the Synchformer env or install ffmpeg."
    out_dir.mkdir(exist_ok=True, parents=True)
    start_ms = int(round(start_sec * 1000))
    dur_ms = int(round(chunk_seconds * 1000))
    safe_stem = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in Path(path).stem)
    new_path = out_dir / f"{safe_stem}_chunk{chunk_index:06d}_{start_ms}ms_{dur_ms}ms.mp4"

    # Re-encode rather than stream-copy so each chunk exactly matches the model's
    # preprocessing assumptions and can be read independently by torchvision.
    #
    # Important: the model transform asks for a fixed number of segments. If the
    # source video is shorter than one chunk, plain `-t chunk_seconds` produces a
    # short file and GenerateMultipleSegments can fail. tpad/apad extend the tail
    # by cloning the last video frame and adding audio silence, then output `-t`
    # trims every generated chunk to exactly chunk_seconds.
    video_filter = (
        f"fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',"
        "crop='trunc(iw/2)'*2:'trunc(ih/2)'*2,"
        f"tpad=stop_mode=clone:stop_duration={chunk_seconds:.3f}"
    )
    audio_filter = f"apad=pad_dur={chunk_seconds:.3f}"
    cmd = [
        which_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "panic",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        path,
        "-vf",
        video_filter,
        "-af",
        audio_filter,
        "-ar",
        str(afps),
        "-ac",
        "1",
        "-t",
        f"{chunk_seconds:.3f}",
        "-avoid_negative_ts",
        "make_zero",
        str(new_path),
    ]
    subprocess.check_call(cmd)

    wav_path = str(new_path).replace(".mp4", ".wav")
    cmd = [
        which_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "panic",
        "-y",
        "-i",
        str(new_path),
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        wav_path,
    ]
    subprocess.call(cmd)
    return str(new_path)


def maybe_reencode_video(path: str, vfps: int, afps: int, in_size: int, out_dir: Path) -> str:
    """Match the preprocessing assumptions used by Synchformer's example.py."""
    video, _, info = torchvision.io.read_video(path, pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"Could not read video frames from {path}")
    _, h, w, _ = video.shape
    video_fps = int(round(float(info.get("video_fps", 0))))
    audio_fps = int(round(float(info.get("audio_fps", 0))))

    needs_reencode = video_fps != vfps or audio_fps != afps or min(h, w) != in_size
    if needs_reencode:
        print(
            f"Reencoding {path}: vfps {video_fps}->{vfps}, afps {audio_fps}->{afps}, "
            f"min_side {min(h, w)}->{in_size}"
        )
        return reencode_video(path, vfps, afps, in_size, out_dir)

    print(f"Skipping reencoding for {path}: vfps={video_fps}, afps={audio_fps}, min_side={min(h, w)}")
    return path


def row_minmax(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mins = x.min(dim=-1, keepdim=True).values
    maxs = x.max(dim=-1, keepdim=True).values
    return (x - mins) / (maxs - mins).clamp_min(eps)


def patch_stage1_cfg(cfg, skip_init_ckpts: bool = True):
    """Keep inference single-process and optionally skip base init checkpoints."""
    cfg.distributed = False
    if "training" not in cfg:
        cfg.training = OmegaConf.create({})
    cfg.training.local_rank = 0
    cfg.training.global_rank = 0
    cfg.training.world_size = 1

    if skip_init_ckpts:
        for tower in ("afeat_extractor", "vfeat_extractor"):
            try:
                cfg.model.params[tower].params.ckpt_path = None
            except Exception:
                pass
    return cfg


def load_checkpoint_and_cfg(checkpoint_path: str, cfg_path: Optional[str]):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if cfg_path is not None:
        cfg = OmegaConf.load(cfg_path)
    else:
        cfg = ckpt.get("args") or ckpt.get("cfg") or ckpt.get("config")
        if cfg is None:
            raise ValueError("No --cfg was provided and no config was found inside the checkpoint.")
        cfg = OmegaConf.create(cfg)
    return ckpt, cfg


def get_state_dict_from_checkpoint(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            if all(torch.is_tensor(v) for v in ckpt.values()):
                state = ckpt
            else:
                raise KeyError("Checkpoint did not contain 'state_dict' or 'model'.")
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(ckpt)}")

    cleaned = {}
    for k, v in state.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def build_batch(video_paths: Iterable[str], cfg, device: torch.device, reencode: bool, out_dir: Path):
    transform = get_transforms(cfg, ["test"])["test"]
    items = []
    used_paths = []
    for path in video_paths:
        path = maybe_reencode_video(path, DEFAULT_VFPS, DEFAULT_AFPS, DEFAULT_SIZE_BEFORE_CROP, out_dir) if reencode else path
        rgb, audio, meta = get_video_and_audio(path, get_meta=True)
        item = {
            "video": rgb,
            "audio": audio,
            "meta": meta,
            "path": path,
            "split": "test",
            "targets": {},
        }
        items.append(transform(item))
        used_paths.append(path)

    batch = torch.utils.data.default_collate(items)
    aud, vid, _ = prepare_inputs(batch, device, get_targets=False)
    return aud, vid, used_paths


def run_forward(model, vid: torch.Tensor, aud: torch.Tensor, cfg, amp: bool, device: torch.device):
    if not hasattr(model, "forward_for_logging"):
        raise AttributeError(
            "The loaded model does not expose forward_for_logging(). Use a Stage-1 AVCLIP checkpoint/config "
            "or adapt this script to call the sync model's feature-extractor path."
        )

    for_loop = bool(cfg.training.get("for_loop_segment_fwd", False))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        with torch.autocast(autocast_device, enabled=amp and device.type == "cuda"):
            out = model.forward_for_logging(vid, aud, for_momentum=False, for_loop=for_loop, do_norm=True)
    return out


def flatten_segment_features(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, S, D] or compatible segment-feature tensors to [B*S, D]."""
    x = x.detach().float()
    if x.ndim < 2:
        raise ValueError(f"Expected feature tensor with at least 2 dims, got shape {tuple(x.shape)}")
    if x.ndim == 2:
        return x
    return x.reshape(-1, x.shape[-1])


def similarities_from_features(vfeat: torch.Tensor, afeat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Build full cosine-similarity matrices from concatenated segment features."""
    vfeat = F.normalize(vfeat.detach().float().cpu(), dim=-1)
    afeat = F.normalize(afeat.detach().float().cpu(), dim=-1)
    return {
        "v2a": vfeat @ afeat.T,
        "a2v": afeat @ vfeat.T,
        "v2v": vfeat @ vfeat.T,
        "a2a": afeat @ afeat.T,
    }


def plot_matrices(
    mats: Dict[str, torch.Tensor],
    out_path: Path,
    scale: str,
    boundaries: Optional[Sequence[int]] = None,
    title_suffix: str = "",
    figsize: float = 10.0,
):
    out_path.parent.mkdir(exist_ok=True, parents=True)
    ordered = ["v2a", "a2v", "v2v", "a2a"]
    fig, axes = plt.subplots(2, 2, figsize=(figsize, figsize))

    for ax, name in zip(axes.flatten(), ordered):
        sim = mats[name].detach().float().cpu()
        if scale == "row_minmax":
            sim = row_minmax(sim)
        elif scale == "none":
            pass
        else:
            raise ValueError(f"Unknown scale: {scale}")

        im = ax.imshow(sim.numpy(), aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("audio segment" if name in {"v2a", "a2a"} else "visual segment")
        ax.set_ylabel("visual segment" if name in {"v2a", "v2v"} else "audio segment")

        if boundaries:
            for boundary_segment in boundaries:
                boundary = boundary_segment - 0.5
                ax.axhline(boundary, linewidth=0.75)
                ax.axvline(boundary, linewidth=0.75)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    title = "Synchformer Stage-1 AVCLIP segment similarity"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def collect_video_paths(vids: Optional[Iterable[str]], video_dir: Optional[str], pattern: str, recursive: bool) -> List[str]:
    paths: List[Path] = []

    if vids:
        paths.extend(Path(v).expanduser().resolve() for v in vids)

    if video_dir is not None:
        root = Path(video_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--video-dir is not a directory: {root}")
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        paths.extend(p.resolve() for p in iterator if p.is_file())

    deduped = sorted(dict.fromkeys(paths))
    if not deduped:
        raise ValueError("No input videos found. Pass --vids or --video-dir with a matching --pattern.")
    return [str(p) for p in deduped]


def make_safe_stem(path: str, index: int) -> str:
    stem = Path(path).stem
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in stem)
    return f"{index:04d}_{safe}"


def iter_transform_objects(obj, seen: Optional[set] = None):
    """Yield nested torchvision/torch transform objects without assuming a type."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)
    yield obj

    children = []
    if hasattr(obj, "transforms"):
        children.extend(list(getattr(obj, "transforms")))
    if isinstance(obj, torch.nn.Module):
        children.extend(list(obj.children()))

    for child in children:
        yield from iter_transform_objects(child, seen)


def infer_segment_window(cfg, vfps: int = DEFAULT_VFPS) -> Optional[Dict[str, float]]:
    """Infer the minimum chunk duration required by GenerateMultipleSegments.

    Synchformer's GenerateMultipleSegments asserts that the input video is long
    enough for `n_segments` segments of `segment_size_vframes`. This function
    finds that transform and computes the minimum number of frames needed using
    the same stride logic as dataset/transforms.py.
    """
    try:
        transform = get_transforms(cfg, ["test"])["test"]
    except Exception as exc:
        print(f"Could not inspect test transforms; falling back to --chunk-seconds or 10s: {exc}")
        return None

    best: Optional[Dict[str, float]] = None
    for obj in iter_transform_objects(transform):
        if not hasattr(obj, "segment_size_vframes"):
            continue
        n_segments = getattr(obj, "n_segments", None)
        if n_segments is None:
            continue
        segment_size_vframes = int(getattr(obj, "segment_size_vframes"))
        step_size_seg = float(getattr(obj, "step_size_seg", 1.0))
        stride_vframes = int(step_size_seg * segment_size_vframes)
        if segment_size_vframes <= 0 or int(n_segments) <= 0 or stride_vframes <= 0:
            continue

        required_frames = segment_size_vframes + (int(n_segments) - 1) * stride_vframes
        # Add one frame of slack so ffmpeg rounding never gives a chunk that is
        # one frame too short. Example: 14*16 frames at 25fps is 8.96s; 9.00s
        # yields 225 frames and safely passes the transform.
        safe_seconds = math.ceil(((required_frames + 1) / vfps) * 1000.0) / 1000.0
        candidate = {
            "required_frames": float(required_frames),
            "safe_seconds": float(safe_seconds),
            "segment_size_vframes": float(segment_size_vframes),
            "n_segments": float(int(n_segments)),
            "step_size_seg": float(step_size_seg),
            "stride_vframes": float(stride_vframes),
        }
        if best is None or candidate["required_frames"] > best["required_frames"]:
            best = candidate
    return best


def resolve_chunk_seconds(args, cfg) -> float:
    """Return a chunk duration that is long enough for the configured transform."""
    inferred = infer_segment_window(cfg, DEFAULT_VFPS)
    fallback = 10.0

    if args.chunk_seconds is None:
        if inferred is not None:
            chunk_seconds = float(inferred["safe_seconds"])
            print(
                "Auto chunk length: "
                f"{chunk_seconds:.3f}s "
                f"({int(inferred['n_segments'])} segments, "
                f"{int(inferred['segment_size_vframes'])} frames/segment, "
                f"step={inferred['step_size_seg']})."
            )
        else:
            chunk_seconds = fallback
            print(f"Auto chunk length unavailable; using fallback {fallback:.3f}s.")
    else:
        chunk_seconds = float(args.chunk_seconds)

    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be > 0")

    if inferred is not None and chunk_seconds + 1e-9 < float(inferred["safe_seconds"]):
        raise ValueError(
            f"--chunk-seconds={chunk_seconds:.3f} is too short for this config. "
            f"Need at least {float(inferred['safe_seconds']):.3f}s "
            f"for {int(inferred['n_segments'])} segments of "
            f"{int(inferred['segment_size_vframes'])} frames "
            f"with step_size_seg={inferred['step_size_seg']}."
        )
    return chunk_seconds


def make_long_video_chunks(
    video_path: str,
    chunk_seconds: float,
    stride_seconds: float,
    chunk_dir: Path,
) -> Tuple[List[str], np.ndarray]:
    """Split one source video into normalized chunks and return paths plus start times."""
    duration = get_media_duration_sec(video_path)
    starts = make_chunk_starts(duration, chunk_seconds, stride_seconds)
    print(
        f"Chunking {video_path}: duration={duration:.2f}s, "
        f"chunk={chunk_seconds:.2f}s, stride={stride_seconds:.2f}s, chunks={len(starts)}"
    )
    chunk_paths = [
        reencode_video_chunk(
            video_path,
            start_sec=start,
            chunk_seconds=chunk_seconds,
            vfps=DEFAULT_VFPS,
            afps=DEFAULT_AFPS,
            in_size=DEFAULT_SIZE_BEFORE_CROP,
            out_dir=chunk_dir,
            chunk_index=i,
        )
        for i, start in enumerate(starts)
    ]
    return chunk_paths, np.array(starts, dtype=np.float32)


def extract_features_for_clips(
    clip_paths: Sequence[str],
    cfg,
    model,
    device: torch.device,
    amp: bool,
    max_clips_per_forward: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Forward clips in mini-batches and concatenate segment features."""
    all_vfeat: List[torch.Tensor] = []
    all_afeat: List[torch.Tensor] = []
    segments_per_clip: Optional[int] = None

    if max_clips_per_forward <= 0:
        raise ValueError("--max-clips-per-forward must be > 0")

    for start in range(0, len(clip_paths), max_clips_per_forward):
        clip_batch = list(clip_paths[start : start + max_clips_per_forward])
        aud, vid, _ = build_batch(clip_batch, cfg, device, reencode=False, out_dir=Path("."))
        out = run_forward(model, vid, aud, cfg, amp=amp, device=device)

        if "segment_vfeat" not in out or "segment_afeat" not in out:
            raise KeyError("forward_for_logging() did not return segment_vfeat and segment_afeat.")

        if segments_per_clip is None:
            segments_per_clip = int(vid.shape[1])
        elif int(vid.shape[1]) != segments_per_clip:
            raise RuntimeError(
                f"Inconsistent segment count per clip: expected {segments_per_clip}, got {int(vid.shape[1])}"
            )

        all_vfeat.append(flatten_segment_features(out["segment_vfeat"]))
        all_afeat.append(flatten_segment_features(out["segment_afeat"]))
        print(f"Forwarded clips {start + 1}-{start + len(clip_batch)} of {len(clip_paths)}")

    if segments_per_clip is None:
        raise ValueError("No clips were provided for feature extraction.")
    return torch.cat(all_vfeat, dim=0), torch.cat(all_afeat, dim=0), segments_per_clip


def process_paths_as_long_video(
    paths: Sequence[str],
    cfg,
    model,
    device: torch.device,
    args,
    plot_path: Path,
    npz_path: Optional[Path],
    work_dir: Path,
    group_label: str,
):
    """Process one or more source videos using chunked full-length extraction."""
    chunk_seconds = float(args.effective_chunk_seconds)
    stride_seconds = args.stride_seconds if args.stride_seconds is not None else chunk_seconds
    all_clip_paths: List[str] = []
    all_chunk_starts: List[float] = []
    source_video_for_chunk: List[int] = []
    video_boundaries_in_clips: List[int] = []

    chunk_dir = work_dir / "chunks"
    for video_index, path in enumerate(paths):
        clip_paths, chunk_starts = make_long_video_chunks(path, chunk_seconds, stride_seconds, chunk_dir / make_safe_stem(path, video_index))
        all_clip_paths.extend(clip_paths)
        all_chunk_starts.extend(float(x) for x in chunk_starts)
        source_video_for_chunk.extend([video_index] * len(clip_paths))
        video_boundaries_in_clips.append(len(all_clip_paths))

    vfeat, afeat, segments_per_clip = extract_features_for_clips(
        all_clip_paths,
        cfg,
        model,
        device=device,
        amp=args.amp,
        max_clips_per_forward=args.max_clips_per_forward,
    )
    mats = similarities_from_features(vfeat, afeat)

    boundaries: List[int] = []
    if args.draw_chunk_boundaries:
        boundaries.extend(i * segments_per_clip for i in range(1, len(all_clip_paths)))
    if len(paths) > 1:
        boundaries.extend(i * segments_per_clip for i in video_boundaries_in_clips[:-1])
    boundaries = sorted(set(boundaries))

    total_segments = int(vfeat.shape[0])
    plot_matrices(
        mats,
        plot_path,
        scale=args.scale,
        boundaries=boundaries,
        title_suffix=f"{group_label}, {total_segments} segments",
        figsize=args.figsize,
    )
    print(f"Saved similarity matrix plot to {plot_path}")

    if npz_path is not None:
        npz_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(
            npz_path,
            **{k: v.detach().float().cpu().numpy() for k, v in mats.items()},
            segment_vfeat=vfeat.detach().float().cpu().numpy(),
            segment_afeat=afeat.detach().float().cpu().numpy(),
            n_segments=np.array([total_segments]),
            segments_per_clip=np.array([segments_per_clip]),
            chunk_starts_sec=np.array(all_chunk_starts, dtype=np.float32),
            chunk_paths=np.array(all_clip_paths, dtype=object),
            source_video_for_chunk=np.array(source_video_for_chunk, dtype=np.int64),
            videos=np.array(list(paths), dtype=object),
            checkpoint=np.array([str(args.checkpoint)], dtype=object),
        )
        print(f"Saved raw matrices/features to {npz_path}")


def process_paths_whole_video(
    paths: Sequence[str],
    cfg,
    model,
    device: torch.device,
    args,
    plot_path: Path,
    npz_path: Optional[Path],
    work_dir: Path,
):
    """Original single-window behavior retained behind --whole-video."""
    aud, vid, used_paths = build_batch(paths, cfg, device, reencode=not args.no_reencode, out_dir=work_dir)
    out = run_forward(model, vid, aud, cfg, amp=args.amp, device=device)
    mats = {
        "v2a": out["segment_sim_v2a"],
        "a2v": out["segment_sim_a2v"],
        "v2v": out["segment_sim_v2v"],
        "a2a": out["segment_sim_a2a"],
    }
    n_videos, n_segments = int(vid.shape[0]), int(vid.shape[1])
    boundaries = [b * n_segments for b in range(1, n_videos)]
    plot_matrices(
        mats,
        plot_path,
        scale=args.scale,
        boundaries=boundaries,
        title_suffix=f"single-window, {n_videos * n_segments} segments",
        figsize=args.figsize,
    )
    print(f"Saved similarity matrix plot to {plot_path}")

    if npz_path is not None:
        npz_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(
            npz_path,
            **{k: v.detach().float().cpu().numpy() for k, v in mats.items()},
            segment_vfeat=out["segment_vfeat"].detach().float().cpu().numpy(),
            segment_afeat=out["segment_afeat"].detach().float().cpu().numpy(),
            n_segments=np.array([n_segments]),
            videos=np.array(used_paths, dtype=object),
            checkpoint=np.array([str(args.checkpoint)], dtype=object),
        )
        print(f"Saved raw matrices/features to {npz_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a Stage-1 AVCLIP .pt checkpoint.")
    parser.add_argument("--cfg", default=None, help="Path to the matching cfg-*.yaml. If omitted, try checkpoint['args'].")
    parser.add_argument("--vids", nargs="+", default=None, help="One or more input .mp4 files.")
    parser.add_argument("--video-dir", default=None, help="Directory containing .mp4 files to process.")
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern used with --video-dir. Default: *.mp4")
    parser.add_argument("--recursive", action="store_true", help="Search --video-dir recursively.")
    parser.add_argument("--combined", action="store_true", help="Plot all videos in one combined block matrix. Default: one PNG per video.")
    parser.add_argument("--out", default="./vis", help="Output PNG path or output directory. With multiple independent videos, this should be a directory.")
    parser.add_argument("--npz", default=None, help="Optional .npz path or directory for raw matrices/features.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-reencode", action="store_true", help="Only affects --whole-video mode; skip ffmpeg normalization.")
    parser.add_argument("--keep-init-ckpts", action="store_true", help="Do not null AST/MotionFormer init ckpt paths before loading checkpoint.")
    parser.add_argument("--scale", choices=["row_minmax", "none"], default="row_minmax", help="How to scale heatmaps before plotting.")
    parser.add_argument("--strict", action="store_true", help="Use strict=True when loading the checkpoint state dict.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for the forward pass.")
    parser.add_argument("--whole-video", action="store_true", help="Use the original single-window behavior instead of chunking.")
    parser.add_argument("--chunk-seconds", type=float, default=None, help="Chunk duration for any-length videos. Default: infer the shortest safe duration from the test transform.")
    parser.add_argument("--stride-seconds", type=float, default=None, help="Stride between chunks. Defaults to --chunk-seconds for non-overlapping coverage.")
    parser.add_argument("--max-clips-per-forward", type=int, default=8, help="Mini-batch size for chunk forward passes.")
    parser.add_argument("--draw-chunk-boundaries", action="store_true", help="Draw grid lines between chunks in the heatmaps.")
    parser.add_argument("--figsize", type=float, default=12.0, help="Matplotlib figure width/height in inches.")
    parser.add_argument("--keep-work", action="store_true", help="Keep temporary work/chunks/reencoded files instead of deleting them after a successful run.")
    args = parser.parse_args()

    video_paths = collect_video_paths(args.vids, args.video_dir, args.pattern, args.recursive)

    device = torch.device(args.device)
    ckpt, cfg = load_checkpoint_and_cfg(args.checkpoint, args.cfg)
    cfg = patch_stage1_cfg(cfg, skip_init_ckpts=not args.keep_init_ckpts)
    args.effective_chunk_seconds = None if args.whole_video else resolve_chunk_seconds(args, cfg)

    _, model = get_model(cfg, device)
    state = get_state_dict_from_checkpoint(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=args.strict)
    if missing:
        print(f"Missing keys while loading checkpoint ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint ({len(unexpected)}): {unexpected[:10]}")
    model.eval()

    out_root = Path(args.out).expanduser().resolve()
    out_is_file = out_root.suffix.lower() == ".png"
    work_dir = (out_root.parent if out_is_file else out_root) / "work"

    try:
        processor = process_paths_whole_video if args.whole_video else process_paths_as_long_video
    
        if args.combined:
            plot_path = out_root if out_is_file else out_root / "combined_similarity.png"
            npz_path = None
            if args.npz is not None:
                npz_root = Path(args.npz).expanduser().resolve()
                npz_path = npz_root if npz_root.suffix.lower() == ".npz" else npz_root / "combined_similarity.npz"
            if args.whole_video:
                processor(video_paths, cfg, model, device, args, plot_path, npz_path, work_dir)
            else:
                processor(video_paths, cfg, model, device, args, plot_path, npz_path, work_dir, "combined")
        else:
            if len(video_paths) == 1 and out_is_file:
                npz_path = Path(args.npz).expanduser().resolve() if args.npz is not None else None
                if args.whole_video:
                    processor(video_paths, cfg, model, device, args, out_root, npz_path, work_dir)
                else:
                    processor(video_paths, cfg, model, device, args, out_root, npz_path, work_dir, Path(video_paths[0]).name)
            else:
                out_root.mkdir(exist_ok=True, parents=True)
                npz_root = Path(args.npz).expanduser().resolve() if args.npz is not None else None
                if npz_root is not None and npz_root.suffix.lower() != ".npz":
                    npz_root.mkdir(exist_ok=True, parents=True)
                for i, video_path in enumerate(video_paths):
                    stem = make_safe_stem(video_path, i)
                    plot_path = out_root / f"{stem}_similarity.png"
                    npz_path = None
                    if npz_root is not None:
                        npz_path = npz_root if (len(video_paths) == 1 and npz_root.suffix.lower() == ".npz") else npz_root / f"{stem}_similarity.npz"
                    if args.whole_video:
                        processor([video_path], cfg, model, device, args, plot_path, npz_path, work_dir)
                    else:
                        processor([video_path], cfg, model, device, args, plot_path, npz_path, work_dir, Path(video_path).name)
    finally:
        if not args.keep_work and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"Deleted temporary work directory {work_dir}")

if __name__ == "__main__":
    main()
