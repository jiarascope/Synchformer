from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def normalize_01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)

def labels_to_rgb(label_map: np.ndarray) -> np.ndarray:
    """
    Fixed palette so colors are stable within one clip.
    """
    palette = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
        ],
        dtype=np.uint8,
    )
    return palette[label_map % len(palette)]

def overlay_rgb(frame_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask_rgb = cv2.resize(
        mask_rgb,
        (frame_rgb.shape[1], frame_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return (frame_rgb * (1.0 - alpha) + mask_rgb * alpha).clip(0, 255).astype(np.uint8)

def eig_to_rgb(eig_maps: torch.Tensor) -> np.ndarray:
    """
    eig_maps: [T,H,W,K]
    Use eigenvectors 1,2,3 as RGB, skipping eig 0.
    """
    eig_np = eig_maps.detach().cpu().numpy()
    if eig_np.shape[-1] >= 4:
        rgb = eig_np[..., 1:4]
    else:
        rgb = eig_np[..., :3]
    rgb = normalize_01(rgb)
    return (255 * rgb).astype(np.uint8)

def pca_features_to_rgb(tokens_grid: torch.Tensor) -> np.ndarray:
    """
    Debug visualization independent of NCut:
    project token features to 3D PCA and make RGB maps.

    tokens_grid: [T,H,W,D]
    returns: [T,H,W,3] uint8
    """
    T, H, W, D = tokens_grid.shape
    X = tokens_grid.reshape(-1, D).detach().cpu().float().numpy()
    X = X - X.mean(axis=0, keepdims=True)

    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)
    Y = normalize_01(Y)
    return (255 * Y.reshape(T, H, W, 3)).astype(np.uint8)

def make_feature_norm_maps(tokens_grid: torch.Tensor) -> np.ndarray:
    """
    tokens_grid: [T,H,W,D]
    returns grayscale RGB maps [T,H,W,3]
    """
    norms = torch.linalg.norm(tokens_grid.float(), dim=-1).detach().cpu().numpy()
    norms = normalize_01(norms)
    rgb = np.repeat(norms[..., None], 3, axis=-1)
    return (255 * rgb).astype(np.uint8)

def write_video(frames_rgb: List[np.ndarray], out_path: Path, fps: float = 8.0):
    if not frames_rgb:
        return

    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for frame_rgb in frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    writer.release()

def token_to_frame_index(t: int, T_tok: int, T_frames: int) -> int:
    """
    Map a token-time index to the nearest representative sampled frame.
    Example: T_tok=8, T_frames=16 gives approx 0,2,4,...,14.
    """
    idx = int(round((t + 0.5) * T_frames / T_tok - 0.5))
    return max(0, min(T_frames - 1, idx))

def add_row_label(img: np.ndarray, text: str) -> np.ndarray:
    """
    img: RGB image row.
    """
    img = img.copy()
    cv2.rectangle(img, (0, 0), (300, 34), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img

def make_contact_sheet(
    frames_rgb: np.ndarray,
    ncut_rgb: np.ndarray,
    labels: np.ndarray,
    pca_rgb: np.ndarray,
    norm_rgb: np.ndarray,
    out_path: Path,
    alpha: float = 0.25,
):
    """
    Rows:
      original representative frames
      feature norm overlay
      PCA feature RGB overlay
      NCut eigen RGB overlay
      NCut cluster overlay
    """
    T_tok = min(ncut_rgb.shape[0], labels.shape[0], pca_rgb.shape[0], norm_rgb.shape[0])
    T_frames = len(frames_rgb)

    rows = [[] for _ in range(5)]

    for t in range(T_tok):
        frame_idx = token_to_frame_index(t, T_tok, T_frames)
        frame = frames_rgb[frame_idx]

        norm_overlay = overlay_rgb(frame, norm_rgb[t], alpha=alpha)
        pca_overlay = overlay_rgb(frame, pca_rgb[t], alpha=alpha)
        ncut_overlay = overlay_rgb(frame, ncut_rgb[t], alpha=alpha)
        cluster_overlay = overlay_rgb(frame, labels_to_rgb(labels[t]), alpha=alpha)

        rows[0].append(frame)
        rows[1].append(norm_overlay)
        rows[2].append(pca_overlay)
        rows[3].append(ncut_overlay)
        rows[4].append(cluster_overlay)

    row_imgs = [
        add_row_label(np.concatenate(rows[0], axis=1), "original representative frames"),
        add_row_label(np.concatenate(rows[1], axis=1), "feature norm overlay"),
        add_row_label(np.concatenate(rows[2], axis=1), "raw feature PCA overlay"),
        add_row_label(np.concatenate(rows[3], axis=1), "NCut eigenvector RGB overlay"),
        add_row_label(np.concatenate(rows[4], axis=1), "NCut k-means cluster overlay"),
    ]

    sheet = np.concatenate(row_imgs, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    # Also save rows separately for easier debugging.
    row_names = [
        "row_0_original.png",
        "row_1_feature_norm.png",
        "row_2_feature_pca.png",
        "row_3_ncut_eig_rgb.png",
        "row_4_ncut_clusters.png",
    ]
    for name, row_img in zip(row_names, row_imgs):
        cv2.imwrite(str(out_path.parent / name), cv2.cvtColor(row_img, cv2.COLOR_RGB2BGR))

def save_feature_debug_plot(tokens_grid: torch.Tensor, out_path: Path):
    """
    Saves a simple per-time-token plot:
      mean feature norm over spatial patches per frame token.
    """
    norms = torch.linalg.norm(tokens_grid.float(), dim=-1)  # [T,H,W]
    mean_norm = norms.mean(dim=(1, 2)).detach().cpu().numpy()
    std_norm = norms.std(dim=(1, 2)).detach().cpu().numpy()

    x = np.arange(len(mean_norm))

    plt.figure(figsize=(8, 4))
    plt.plot(x, mean_norm, marker="o")
    plt.fill_between(x, mean_norm - std_norm, mean_norm + std_norm, alpha=0.2)
    plt.xlabel("Time token")
    plt.ylabel("Feature norm")
    plt.title("Motionformer token feature norm by time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Main processing
# -----------------------------
