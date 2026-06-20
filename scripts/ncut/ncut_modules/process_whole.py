from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .process_common import (
    accumulate_overlay_frames,
    effective_duration,
    extract_window_segments,
    make_window_starts,
    records_without_images,
    run_embedding_pipeline,
    write_held_overlay_frame_videos,
    write_json,
)
from .video_io import get_video_info
from .viz import add_row_label, labels_to_rgb, overlay_rgb, token_to_frame_index


def process_whole_video(
    video_path: Path,
    out_root: Path,
    visual_encoder: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
):
    """
    Slide a MotionFormer window across the whole video, collect all
    spatiotemporal tokens, run NCut globally, and write whole-video overlays.
    """
    stem = video_path.stem
    out_dir = out_root / stem / "whole_video"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    video_fps = info["fps"]
    duration = effective_duration(info, args.max_duration_sec)
    starts = make_window_starts(duration, args.segment_sec, args.stride_sec)

    print(f"[whole] video fps={video_fps:.3f}, duration={info['duration_sec']:.3f}s")
    print(f"[whole] using duration={duration:.3f}s")
    print(f"[whole] num windows={len(starts)}")
    print(f"[whole] segment_sec={args.segment_sec}, stride_sec={args.stride_sec}")

    # ------------------------------------------------------------------
    # 1. Extract tokens for each sliding segment.
    # ------------------------------------------------------------------
    extraction = extract_window_segments(
        video_path=video_path,
        starts=starts,
        visual_encoder=visual_encoder,
        device=device,
        args=args,
        log_prefix="whole",
    )
    segment_records = extraction.records
    X_all = extraction.tokens_flat

    print(f"[whole] X_all before normalize: {tuple(X_all.shape)}")
    torch.save(X_all, out_dir / "whole_video_tokens_flat.pt")

    # ------------------------------------------------------------------
    # 2. Run NCut globally over all tokens.
    # ------------------------------------------------------------------
    if args.ncut_mode != "global":
        raise NotImplementedError("For now use --ncut_mode global")

    print("[whole] running global NCut...")
    eig_all, labels_all, rgb_all = run_embedding_pipeline(
        X_all,
        args=args,
        device=device,
        rgb=True,
    )
    assert rgb_all is not None

    print(
        f"[whole] computed UMAP color from first "
        f"{min(args.eig_rgb_dims, eig_all.shape[1])} eigenvectors"
    )

    np.save(out_dir / "whole_video_umap_rgb_flat.npy", rgb_all)
    torch.save(eig_all, out_dir / "whole_video_ncut_eig_flat.pt")
    np.save(out_dir / "whole_video_cluster_labels_flat.npy", labels_all)

    # ------------------------------------------------------------------
    # 3. Accumulate overlays onto representative global frames.
    # ------------------------------------------------------------------
    total_frames = info["total_frames"]
    overlay_accum_ncut, overlay_accum_cluster, overlay_count = accumulate_overlay_frames(
        records=segment_records,
        rgb_flat=rgb_all,
        labels_flat=labels_all,
        total_frames=total_frames,
        alpha=args.alpha,
    )

    contact_rows_original = []
    contact_rows_ncut = []
    contact_rows_cluster = []

    for rec in segment_records:
        wi = rec["window_index"]
        offset = rec["token_offset"]
        n_tokens = rec["num_tokens"]
        T_tok, H_tok, W_tok, _ = rec["tokens_shape"]

        ncut_rgb = rgb_all[offset : offset + n_tokens].reshape(T_tok, H_tok, W_tok, 3)
        labels_seg = labels_all[offset : offset + n_tokens].reshape(T_tok, H_tok, W_tok)
        cluster_rgb = labels_to_rgb(labels_seg)

        vis_frames = rec["vis_frames"]
        sampled_indices = rec["sampled_indices"]

        # For contact sheet, save only a few windows/tokens.
        for t in range(T_tok):
            if wi >= 4 or t not in {0, T_tok // 2, T_tok - 1}:
                continue

            local_frame_idx = token_to_frame_index(t, T_tok, len(sampled_indices))
            frame = vis_frames[local_frame_idx]

            contact_rows_original.append(frame)
            contact_rows_ncut.append(overlay_rgb(frame, ncut_rgb[t], alpha=args.alpha))
            contact_rows_cluster.append(overlay_rgb(frame, cluster_rgb[t], alpha=args.alpha))

    # ------------------------------------------------------------------
    # 4. Decode original video and write full overlay videos.
    # ------------------------------------------------------------------
    out_size = (args.image_size, args.image_size)
    write_held_overlay_frame_videos(
        video_path=video_path,
        out_paths={
            "ncut": out_dir / "whole_ncut_rgb_overlay.mp4",
            "cluster": out_dir / "whole_ncut_clusters_overlay.mp4",
        },
        accumulators={
            "ncut": overlay_accum_ncut,
            "cluster": overlay_accum_cluster,
        },
        counts=overlay_count,
        video_fps=video_fps,
        out_size=out_size,
        max_duration_sec=args.max_duration_sec,
    )

    # ------------------------------------------------------------------
    # 5. Contact sheet summary.
    # ------------------------------------------------------------------
    if contact_rows_original:
        row0 = add_row_label(np.concatenate(contact_rows_original, axis=1), "original sampled frames")
        row1 = add_row_label(np.concatenate(contact_rows_ncut, axis=1), "global NCut eigen RGB")
        row2 = add_row_label(np.concatenate(contact_rows_cluster, axis=1), "global NCut clusters")

        sheet = np.concatenate([row0, row1, row2], axis=0)
        cv2.imwrite(str(out_dir / "whole_video_contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    write_json(
        out_dir / "whole_video_segments.json",
        {
            "video": str(video_path),
            "video_info": info,
            "segment_sec": args.segment_sec,
            "stride_sec": args.stride_sec,
            "num_windows": len(starts),
            "segments": records_without_images(segment_records),
        },
    )

    print(f"[whole] wrote outputs to: {out_dir}")
