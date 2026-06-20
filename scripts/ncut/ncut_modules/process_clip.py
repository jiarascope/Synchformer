from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .process_common import (
    cluster_embedding,
    flatten_tokens,
    load_clip_tokens,
    run_ncut_embedding,
    write_json,
)
from .viz import (
    eig_to_rgb,
    labels_to_rgb,
    make_contact_sheet,
    make_feature_norm_maps,
    overlay_rgb,
    pca_features_to_rgb,
    save_feature_debug_plot,
    token_to_frame_index,
    write_video,
)


def process_one_video(
    video_path: Path,
    out_root: Path,
    visual_encoder: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
):
    stem = video_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result = load_clip_tokens(
        video_path=video_path,
        visual_encoder=visual_encoder,
        device=device,
        args=args,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )
    clip = result.clip
    vis_frames = result.vis_frames
    meta = result.meta
    tokens_grid = result.tokens_grid

    # Save raw/debug features.
    torch.save(tokens_grid.detach().cpu(), out_dir / "features_tokens_grid.pt")

    frame_pooled = tokens_grid.mean(dim=(1, 2))
    torch.save(frame_pooled.detach().cpu(), out_dir / "features_frame_pooled.pt")

    write_json(out_dir / "meta.json", meta)

    print(f"[{stem}] clip tensor: {tuple(clip.shape)}")
    print(f"[{stem}] tokens_grid: {tuple(tokens_grid.shape)}")
    print(f"[{stem}] frame_pooled: {tuple(frame_pooled.shape)}")

    # Feature debugging visualizations before NCut.
    norm_rgb = make_feature_norm_maps(tokens_grid)
    pca_rgb = pca_features_to_rgb(tokens_grid)
    save_feature_debug_plot(tokens_grid, out_dir / "feature_norms.png")

    # Run NCut.
    T, H, W, _ = tokens_grid.shape
    eig = run_ncut_embedding(
        flatten_tokens(tokens_grid),
        num_eig=args.num_eig,
        device=device,
    )

    torch.save(eig, out_dir / "ncut_eig_flat.pt")

    eig_maps = eig.reshape(T, H, W, -1)
    torch.save(eig_maps, out_dir / "ncut_eig_maps.pt")

    ncut_rgb = eig_to_rgb(eig_maps)

    labels = cluster_embedding(
        eig,
        num_eig=args.num_eig,
        kmeans_dims=args.kmeans_dims,
        num_clusters=args.num_clusters,
        seed=args.seed,
    ).reshape(T, H, W)
    np.save(out_dir / "ncut_cluster_labels.npy", labels)

    make_contact_sheet(
        frames_rgb=vis_frames,
        ncut_rgb=ncut_rgb,
        labels=labels,
        pca_rgb=pca_rgb,
        norm_rgb=norm_rgb,
        out_path=out_dir / "contact_sheet.png",
        alpha=args.alpha,
    )

    # MP4 overlays.
    T_tok = ncut_rgb.shape[0]
    T_frames = len(vis_frames)

    ncut_overlay_frames = []
    cluster_overlay_frames = []
    pca_overlay_frames = []

    for t in range(T_tok):
        frame_idx = token_to_frame_index(t, T_tok, T_frames)
        frame = vis_frames[frame_idx]

        ncut_overlay_frames.append(
            overlay_rgb(frame, ncut_rgb[t], alpha=args.alpha)
        )
        cluster_overlay_frames.append(
            overlay_rgb(frame, labels_to_rgb(labels[t]), alpha=args.alpha)
        )
        pca_overlay_frames.append(
            overlay_rgb(frame, pca_rgb[t], alpha=args.alpha)
        )

    write_video(ncut_overlay_frames, out_dir / "ncut_rgb_overlay.mp4", fps=args.out_fps)
    write_video(cluster_overlay_frames, out_dir / "ncut_clusters_overlay.mp4", fps=args.out_fps)
    write_video(pca_overlay_frames, out_dir / "feature_pca_rgb_overlay.mp4", fps=args.out_fps)

    print(f"[{stem}] wrote outputs to: {out_dir}")
