# scripts/synchformer_clip_preprocess.py

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple, Dict, Any

import cv2
import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _center_crop_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    """
    frame: RGB uint8, [H, W, 3]
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    frame = frame[y0:y0 + side, x0:x0 + side]
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return frame


def _resize_short_side_then_center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """
    Resize so short side == size, then center crop size x size.
    frame: RGB uint8, [H, W, 3]
    """
    h, w = frame.shape[:2]
    if h < w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))

    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    y0 = max((new_h - size) // 2, 0)
    x0 = max((new_w - size) // 2, 0)
    return frame[y0:y0 + size, x0:x0 + size]


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = 8,
    sampling: Literal["uniform", "first", "center"] = "uniform",
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decode and sample RGB frames from a video.

    Returns:
        frames_rgb: uint8 array, [T, H, W, 3]
        meta: fps, sampled frame indices, etc.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not determine frame count for: {video_path}")

    lo = 0
    hi = total - 1

    if start_sec is not None:
        lo = max(0, int(round(start_sec * fps)))

    if duration_sec is not None:
        hi = min(total - 1, lo + int(round(duration_sec * fps)) - 1)

    if lo > hi:
        raise ValueError(
            f"Invalid sampling range: lo={lo}, hi={hi}, "
            f"start_sec={start_sec}, duration_sec={duration_sec}, fps={fps}"
        )

    available = hi - lo + 1

    if sampling == "uniform":
        if available >= num_frames:
            indices = np.linspace(lo, hi, num_frames).round().astype(int)
        else:
            # Repeat last frame if clip is too short.
            indices = np.linspace(lo, hi, available).round().astype(int)
            pad = np.full(num_frames - available, indices[-1], dtype=int)
            indices = np.concatenate([indices, pad])

    elif sampling == "first":
        indices = np.arange(lo, min(lo + num_frames, hi + 1), dtype=int)
        if len(indices) < num_frames:
            pad = np.full(num_frames - len(indices), indices[-1], dtype=int)
            indices = np.concatenate([indices, pad])

    elif sampling == "center":
        center = (lo + hi) // 2
        half = num_frames // 2
        start = max(lo, center - half)
        end = min(hi + 1, start + num_frames)
        indices = np.arange(start, end, dtype=int)
        if len(indices) < num_frames:
            pad = np.full(num_frames - len(indices), indices[-1], dtype=int)
            indices = np.concatenate([indices, pad])

    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    frames_rgb = np.stack(frames, axis=0)

    meta = {
        "video_path": str(video_path),
        "fps": float(fps),
        "total_frames": int(total),
        "sampled_indices": indices.tolist(),
        "sampled_times_sec": (indices / fps).tolist(),
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "sampling": sampling,
    }
    return frames_rgb, meta


def preprocess_frames_for_motionformer(
    frames_rgb: np.ndarray,
    size: int = 224,
    crop_mode: Literal["resize_short_side", "square_center_crop"] = "resize_short_side",
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Convert RGB uint8 frames to Motionformer/Synchformer-style tensor.

    Args:
        frames_rgb: uint8, [T, H, W, 3]

    Returns:
        clip_tensor: float32, [1, 3, T, size, size]
        vis_frames: uint8, [T, size, size, 3], resized/cropped but unnormalized
    """
    assert frames_rgb.ndim == 4 and frames_rgb.shape[-1] == 3, frames_rgb.shape

    processed = []
    for frame in frames_rgb:
        if crop_mode == "resize_short_side":
            frame = _resize_short_side_then_center_crop(frame, size)
        elif crop_mode == "square_center_crop":
            frame = _center_crop_rgb(frame, size)
        else:
            raise ValueError(f"Unknown crop_mode: {crop_mode}")
        processed.append(frame)

    vis_frames = np.stack(processed, axis=0).astype(np.uint8)

    arr = vis_frames.astype(np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 3)
    arr = (arr - mean) / std

    # [T, H, W, C] -> [1, C, T, H, W]
    clip_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return clip_tensor.float(), vis_frames


def load_and_preprocess_clip(
    video_path: str | Path,
    num_frames: int = 8,
    size: int = 224,
    sampling: Literal["uniform", "first", "center"] = "uniform",
    crop_mode: Literal["resize_short_side", "square_center_crop"] = "resize_short_side",
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """
    Convenience wrapper.

    Returns:
        clip_tensor: [1, 3, T, H, W]
        vis_frames:  [T, H, W, 3], uint8
        meta: dict
    """
    raw_frames, meta = sample_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        sampling=sampling,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    clip_tensor, vis_frames = preprocess_frames_for_motionformer(
        raw_frames,
        size=size,
        crop_mode=crop_mode,
    )
    return clip_tensor, vis_frames, meta