from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


def rasterize_mask(
    coords: np.ndarray,
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    spec_shape_tf: Tuple[int, int],
    desc: str = "Rasterizing mask",
    ignore_label: int | None = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize patch colors to spectrogram pixels. Returns rgb(F,T,3), coverage(F,T).

    Cluster-specific/recursive NCut uses -1 for tokens that were not selected
    for the second pass. Those labels are intentionally left grayscale.
    """
    T, Freq = spec_shape_tf
    acc = np.zeros((Freq, T, 3), dtype=np.float32)
    cnt = np.zeros((Freq, T, 1), dtype=np.float32)
    for (f0, f1, t0, t1), c_raw in tqdm(zip(coords, clusters), total=len(clusters), desc=desc):
        c = int(c_raw)
        if ignore_label is not None and c == int(ignore_label):
            continue
        if c < 0:
            continue
        if c >= int(cluster_rgb.shape[0]):
            raise IndexError(f"Cluster label {c} has no color; cluster_rgb has {cluster_rgb.shape[0]} rows")
        color = cluster_rgb[c]
        acc[f0:f1, t0:t1, :] += color
        cnt[f0:f1, t0:t1, :] += 1.0
    rgb = acc / np.maximum(cnt, 1.0)
    coverage = (cnt[..., 0] > 0).astype(np.float32)
    return rgb, coverage


def make_overlay_image(
    fbank: torch.Tensor,
    mask_rgb: np.ndarray,
    coverage: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return RGB uint8 image as (F,T,3), low frequency at bottom after later flip."""
    spec = fbank.cpu().numpy().T  # (F, T)
    lo, hi = np.percentile(spec, [1, 99])
    gray = np.clip((spec - lo) / max(hi - lo, 1e-6), 0, 1)
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    a = (alpha * coverage[..., None]).astype(np.float32)
    out = (1.0 - a) * base + a * mask_rgb
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)




def write_global_grid_image(
    overlays: Sequence[Tuple[Path, np.ndarray]],
    out_png: Path,
    row_width: int = 1600,
    row_height: int = 220,
    label_width: int = 320,
    gap: int = 8,
) -> None:
    """Write a PNG montage of all per-video NCut spectrogram overlays."""
    if not overlays:
        return
    rows = []
    for path, overlay_ft in overlays:
        spec_rgb = cv2.resize(np.flipud(overlay_ft), (row_width, row_height), interpolation=cv2.INTER_AREA)
        row = np.full((row_height, label_width + row_width + gap, 3), 18, dtype=np.uint8)
        row[:, label_width + gap:] = spec_rgb[:, :, ::-1]
        cv2.putText(row, path.name[:38], (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(row, f"frames: {overlay_ft.shape[1]}", (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        rows.append(row)
    canvas_h = len(rows) * row_height + (len(rows) - 1) * gap
    canvas_w = label_width + row_width + gap
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
    y = 0
    for row in rows:
        canvas[y:y + row_height] = row
        y += row_height + gap
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)
    print(f"Wrote global grid image: {out_png}")


