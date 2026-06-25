from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from preprocess import load_and_preprocess_clip

from .model import extract_spatiotemporal_tokens
from .ncut_ops import run_ncut
from .viz import labels_to_rgb, overlay_rgb, token_to_frame_index


@dataclass
class ClipTokenResult:
    """Outputs from one load/preprocess/token-extraction pass."""

    clip: torch.Tensor
    vis_frames: np.ndarray
    meta: Dict[str, Any]
    tokens_grid: torch.Tensor


@dataclass
class SegmentExtractionResult:
    """Flattened token matrix plus per-window metadata records."""

    tokens_flat: torch.Tensor
    records: List[Dict[str, Any]]
    next_token_offset: int


# -----------------------------------------------------------------------------
# Clip/window extraction
# -----------------------------------------------------------------------------


def load_clip_tokens(
    *,
    video_path: Path,
    visual_encoder: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    start_sec: Optional[float],
    duration_sec: Optional[float],
) -> ClipTokenResult:
    """
    Load one video clip/window, move it to the requested device, adapt it to
    Synchformer's expected shape, and extract MotionFormer patch tokens.
    """
    clip, vis_frames, meta = load_and_preprocess_clip(
        video_path=video_path,
        num_frames=args.num_frames,
        size=args.image_size,
        sampling=args.sampling,
        crop_mode=args.crop_mode,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )

    clip = clip.to(device)

    # Synchformer MotionFormer expects [B, S, C, T, H, W].
    if clip.ndim == 5:
        clip = clip.unsqueeze(1)

    tokens_grid = extract_spatiotemporal_tokens(
        visual_encoder=visual_encoder,
        clip=clip,
        num_frames=args.num_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        feature_key=args.feature_key,
    )

    return ClipTokenResult(
        clip=clip,
        vis_frames=vis_frames,
        meta=meta,
        tokens_grid=tokens_grid,
    )


def flatten_tokens(tokens_grid: torch.Tensor) -> torch.Tensor:
    """Convert [T,H,W,D] tokens into a detached CPU [N,D] matrix."""
    T, H, W, D = tokens_grid.shape
    return tokens_grid.reshape(T * H * W, D).detach().cpu()


def make_window_starts(duration_sec: float, segment_sec: float, stride_sec: float) -> List[float]:
    """Return sliding-window starts, preserving the original fallback to [0.0]."""
    starts: List[float] = []
    s = 0.0
    while s + segment_sec <= duration_sec + 1e-6:
        starts.append(s)
        s += stride_sec

    if len(starts) == 0:
        starts = [0.0]

    return starts


def effective_duration(info: Mapping[str, Any], max_duration_sec: Optional[float]) -> float:
    """Apply --max_duration_sec to video metadata duration."""
    duration = float(info["duration_sec"])
    if max_duration_sec is not None:
        duration = min(duration, max_duration_sec)
    return duration


def extract_window_segments(
    *,
    video_path: Path,
    starts: Iterable[float],
    visual_encoder: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    log_prefix: str,
    record_prefix: Optional[Mapping[str, Any]] = None,
    initial_token_offset: int = 0,
) -> SegmentExtractionResult:
    """
    Extract tokens for each start time and build records with the common fields
    consumed by both whole-video and joint-directory processing.
    """
    starts = list(starts)
    all_tokens: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []
    token_offset = int(initial_token_offset)
    record_prefix = dict(record_prefix or {})

    for wi, start_sec in enumerate(starts):
        print(f"[{log_prefix}] extracting window {wi + 1}/{len(starts)} @ {start_sec:.3f}s")

        result = load_clip_tokens(
            video_path=video_path,
            visual_encoder=visual_encoder,
            device=device,
            args=args,
            start_sec=start_sec,
            duration_sec=args.segment_sec,
        )

        T_tok, H_tok, W_tok, D = result.tokens_grid.shape
        flat = flatten_tokens(result.tokens_grid)
        all_tokens.append(flat)

        rec: Dict[str, Any] = dict(record_prefix)
        rec.update(
            {
                "window_index": wi,
                "start_sec": float(start_sec),
                "token_offset": int(token_offset),
                "num_tokens": int(flat.shape[0]),
                "tokens_shape": [int(T_tok), int(H_tok), int(W_tok), int(D)],
                "sampled_indices": result.meta["sampled_indices"],
                "sampled_times_sec": result.meta["sampled_times_sec"],
                # Keep images in memory only; strip before JSON serialization.
                "vis_frames": result.vis_frames,
            }
        )
        records.append(rec)
        token_offset += int(flat.shape[0])

    if not all_tokens:
        raise RuntimeError(f"No token windows were extracted for: {video_path}")

    return SegmentExtractionResult(
        tokens_flat=torch.cat(all_tokens, dim=0),
        records=records,
        next_token_offset=token_offset,
    )


# -----------------------------------------------------------------------------
# NCut, UMAP color, and clustering
# -----------------------------------------------------------------------------


def run_ncut_embedding(
    tokens_flat: torch.Tensor,
    *,
    num_eig: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize token features, run NCut, and return CPU eigenvectors."""
    X = F.normalize(tokens_flat.float(), dim=-1)
    eig = run_ncut(X.to(device), num_eig=num_eig, device=device)
    return eig.detach().cpu()


def _rgb_float_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Normalize or scale color coordinates into uint8 RGB."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() <= 1.0 and rgb.min() >= 0.0:
        rgb = rgb * 255.0
    elif rgb.max() <= 255.0 and rgb.min() >= 0.0:
        rgb = rgb
    else:
        rgb = rgb - rgb.min(axis=0, keepdims=True)
        denom = rgb.max(axis=0, keepdims=True)
        rgb = np.divide(rgb, denom + 1e-6) * 255.0

    return rgb.clip(0, 255).astype(np.uint8)


def embedding_to_umap_rgb(eig: torch.Tensor, eig_rgb_dims: int) -> np.ndarray:
    """Map NCut eigenvectors to RGB using ncut_pytorch's UMAP coloring."""
    from ncut_pytorch.color import umap_color

    eig_dims = min(eig_rgb_dims, eig.shape[1])
    rgb = umap_color(eig[:, :eig_dims])

    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    # ncut_pytorch color helpers may return either 0..1 floats or 0..255 values.
    return _rgb_float_to_uint8(rgb)


def embedding_to_tsne_rgb(eig: torch.Tensor, eig_rgb_dims: int, seed: int) -> np.ndarray:
    """Map NCut eigenvectors to RGB using 3D t-SNE coordinates."""
    from sklearn.manifold import TSNE

    eig_dims = min(eig_rgb_dims, eig.shape[1])
    X = eig[:, :eig_dims].detach().cpu().float().numpy()

    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 3), dtype=np.uint8)

    perplexity = min(30.0, max(1.0, (X.shape[0] - 1) / 3.0))
    rgb = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X)

    return _rgb_float_to_uint8(rgb)


def embedding_to_rgb(eig: torch.Tensor, *, args: argparse.Namespace) -> np.ndarray:
    """Map NCut eigenvectors to RGB using the requested reducer."""
    mapping = getattr(args, "embedding_map", "umap")
    if mapping == "umap":
        return embedding_to_umap_rgb(eig, args.eig_rgb_dims)
    if mapping == "tsne":
        return embedding_to_tsne_rgb(eig, args.eig_rgb_dims, args.seed)
    raise ValueError(f"Unsupported embedding_map: {mapping}")


def cluster_embedding(
    eig: torch.Tensor,
    *,
    num_eig: int,
    kmeans_dims: int,
    num_clusters: int,
    seed: int,
) -> np.ndarray:
    """Run the shared k-means step used by all process modules."""
    Z = eig[:, 1 : min(num_eig, kmeans_dims + 1)].numpy()
    return KMeans(
        n_clusters=num_clusters,
        n_init="auto",
        random_state=seed,
    ).fit_predict(Z)


def run_embedding_pipeline(
    tokens_flat: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    rgb: bool = False,
) -> Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
    """
    Run the common NCut+k-means pipeline, optionally also returning mapped RGB.

    Returns:
        eig: CPU tensor [N,K]
        labels: numpy array [N]
        rgb_flat: optional uint8 array [N,3]
    """
    eig = run_ncut_embedding(tokens_flat, num_eig=args.num_eig, device=device)
    labels = cluster_embedding(
        eig,
        num_eig=args.num_eig,
        kmeans_dims=args.kmeans_dims,
        num_clusters=args.num_clusters,
        seed=args.seed,
    )
    rgb_flat = embedding_to_rgb(eig, args=args) if rgb else None
    return eig, labels, rgb_flat


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------


def records_without_images(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Copy segment records and remove in-memory visualization frames."""
    clean_records: List[Dict[str, Any]] = []
    for rec in records:
        r = dict(rec)
        r.pop("vis_frames", None)
        clean_records.append(r)
    return clean_records


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def group_records_by_key(records: Iterable[Mapping[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec[key], []).append(dict(rec))
    return grouped


# -----------------------------------------------------------------------------
# Overlay accumulation/writing helpers
# -----------------------------------------------------------------------------


def _add_to_accumulator(
    accum: MutableMapping[int, np.ndarray],
    count: MutableMapping[int, int],
    frame_idx: int,
    image: np.ndarray,
) -> None:
    if frame_idx not in accum:
        accum[frame_idx] = image.astype(np.float32)
        count[frame_idx] = 1
    else:
        accum[frame_idx] += image.astype(np.float32)
        count[frame_idx] += 1


def accumulate_overlay_frames(
    *,
    records: Iterable[Mapping[str, Any]],
    rgb_flat: np.ndarray,
    labels_flat: np.ndarray,
    total_frames: int,
    alpha: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int]]:
    """
    Whole-video path: create overlay frames for sampled positions and average
    overlapping windows at the same global frame index.
    """
    ncut_accum: Dict[int, np.ndarray] = {}
    cluster_accum: Dict[int, np.ndarray] = {}
    count: Dict[int, int] = {}

    for rec in records:
        offset = int(rec["token_offset"])
        n_tokens = int(rec["num_tokens"])
        T_tok, H_tok, W_tok, _ = rec["tokens_shape"]

        rgb_seg = rgb_flat[offset : offset + n_tokens].reshape(T_tok, H_tok, W_tok, 3)
        labels_seg = labels_flat[offset : offset + n_tokens].reshape(T_tok, H_tok, W_tok)
        cluster_rgb = labels_to_rgb(labels_seg)

        vis_frames = rec["vis_frames"]
        sampled_indices = rec["sampled_indices"]

        for t in range(T_tok):
            local_frame_idx = token_to_frame_index(t, T_tok, len(sampled_indices))
            global_frame_idx = int(sampled_indices[local_frame_idx])

            if global_frame_idx < 0 or global_frame_idx >= total_frames:
                continue

            frame = vis_frames[local_frame_idx]
            ncut_overlay = overlay_rgb(frame, rgb_seg[t], alpha=alpha)
            cluster_overlay = overlay_rgb(frame, cluster_rgb[t], alpha=alpha)

            _add_to_accumulator(ncut_accum, count, global_frame_idx, ncut_overlay)

            # Keep separate sums but shared counts.
            if global_frame_idx not in cluster_accum:
                cluster_accum[global_frame_idx] = cluster_overlay.astype(np.float32)
            else:
                cluster_accum[global_frame_idx] += cluster_overlay.astype(np.float32)

    return ncut_accum, cluster_accum, count


def write_held_overlay_frame_videos(
    *,
    video_path: Path,
    out_paths: Mapping[str, Path],
    accumulators: Mapping[str, Mapping[int, np.ndarray]],
    counts: Mapping[int, int],
    video_fps: float,
    out_size: Tuple[int, int],
    max_duration_sec: Optional[float],
) -> None:
    """
    Decode a video and write held overlay frames. When a sampled overlay is
    missing for an output frame, the previous overlay is reused.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        name: cv2.VideoWriter(str(path), fourcc, video_fps, out_size)
        for name, path in out_paths.items()
    }
    last_frames: Dict[str, Optional[np.ndarray]] = {name: None for name in out_paths}

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if max_duration_sec is not None and frame_idx / video_fps > max_duration_sec:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, out_size, interpolation=cv2.INTER_AREA)

        for name, writer in writers.items():
            accum = accumulators[name]
            if frame_idx in accum:
                n = counts[frame_idx]
                last_frames[name] = (accum[frame_idx] / n).clip(0, 255).astype(np.uint8)

            out_frame = last_frames[name] if last_frames[name] is not None else frame_rgb
            writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

        frame_idx += 1

    cap.release()
    for writer in writers.values():
        writer.release()


def accumulate_masks_for_video(
    *,
    records: Iterable[Mapping[str, Any]],
    rgb_flat: np.ndarray,
    labels_flat: np.ndarray,
    total_frames: int,
    out_size: Tuple[int, int],
    num_clusters: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int]]:
    """
    Joint-directory path: accumulate continuous RGB masks and cluster
    probabilities at sampled global frame indices.
    """
    rgb_mask_accum: Dict[int, np.ndarray] = {}
    cluster_prob_accum: Dict[int, np.ndarray] = {}
    mask_count: Dict[int, int] = {}

    for rec in records:
        offset = int(rec["token_offset"])
        n_tokens = int(rec["num_tokens"])
        T_tok, H_tok, W_tok, _ = rec["tokens_shape"]

        rgb_seg = rgb_flat[offset : offset + n_tokens].reshape(T_tok, H_tok, W_tok, 3)
        labels_seg = labels_flat[offset : offset + n_tokens].reshape(T_tok, H_tok, W_tok)
        sampled_indices = rec["sampled_indices"]

        for t in range(T_tok):
            local_frame_idx = token_to_frame_index(t, T_tok, len(sampled_indices))
            global_frame_idx = int(sampled_indices[local_frame_idx])

            if global_frame_idx < 0 or global_frame_idx >= total_frames:
                continue

            # Continuous UMAP RGB mask: use smooth interpolation.
            rgb_mask = cv2.resize(
                rgb_seg[t],
                out_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)

            # Discrete clusters: accumulate probabilities, not colors.
            onehot = np.eye(num_clusters, dtype=np.float32)[labels_seg[t]]
            onehot = cv2.resize(
                onehot,
                out_size,
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)

            if global_frame_idx not in rgb_mask_accum:
                rgb_mask_accum[global_frame_idx] = rgb_mask
                cluster_prob_accum[global_frame_idx] = onehot
                mask_count[global_frame_idx] = 1
            else:
                rgb_mask_accum[global_frame_idx] += rgb_mask
                cluster_prob_accum[global_frame_idx] += onehot
                mask_count[global_frame_idx] += 1

    return rgb_mask_accum, cluster_prob_accum, mask_count


def write_held_mask_overlay_videos(
    *,
    video_path: Path,
    rgb_out_path: Path,
    cluster_out_path: Path,
    rgb_mask_accum: Mapping[int, np.ndarray],
    cluster_prob_accum: Mapping[int, np.ndarray],
    mask_count: Mapping[int, int],
    video_fps: float,
    out_size: Tuple[int, int],
    max_duration_sec: Optional[float],
    alpha: float,
) -> None:
    """Decode a video and overlay the latest accumulated masks on each frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb_writer = cv2.VideoWriter(str(rgb_out_path), fourcc, video_fps, out_size)
    cluster_writer = cv2.VideoWriter(str(cluster_out_path), fourcc, video_fps, out_size)

    frame_idx = 0
    last_rgb_mask: Optional[np.ndarray] = None
    last_cluster_rgb: Optional[np.ndarray] = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if max_duration_sec is not None and frame_idx / video_fps > max_duration_sec:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, out_size, interpolation=cv2.INTER_AREA)

        if frame_idx in rgb_mask_accum:
            n = mask_count[frame_idx]
            last_rgb_mask = (rgb_mask_accum[frame_idx] / n).clip(0, 255).astype(np.uint8)

            cluster_probs = cluster_prob_accum[frame_idx] / n
            cluster_labels = cluster_probs.argmax(axis=-1)
            last_cluster_rgb = labels_to_rgb(cluster_labels)

        rgb_frame = (
            overlay_rgb(frame_rgb, last_rgb_mask, alpha=alpha)
            if last_rgb_mask is not None
            else frame_rgb
        )
        cluster_frame = (
            overlay_rgb(frame_rgb, last_cluster_rgb, alpha=alpha)
            if last_cluster_rgb is not None
            else frame_rgb
        )

        rgb_writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        cluster_writer.write(cv2.cvtColor(cluster_frame, cv2.COLOR_RGB2BGR))

        frame_idx += 1

    cap.release()
    rgb_writer.release()
    cluster_writer.release()
