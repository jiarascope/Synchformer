from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .process_common import (
    accumulate_masks_for_video,
    effective_duration,
    extract_window_segments,
    group_records_by_key,
    make_window_starts,
    records_without_images,
    run_embedding_pipeline,
    write_held_mask_overlay_videos,
    write_json,
)
from .video_io import get_video_info, iter_videos


def process_video_directory_joint_ncut(
    video_dir: Path,
    out_root: Path,
    visual_encoder: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
):
    """
    Extract MotionFormer tokens from all videos in video_dir, concatenate them,
    run ONE NCut over all tokens, then write per-video shared-color overlays.
    """
    out_dir = out_root / "joint_video_dir_ncut"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = list(iter_videos(None, str(video_dir)))
    if not video_paths:
        raise RuntimeError(f"No videos found in: {video_dir}")

    print(f"[joint] found {len(video_paths)} videos")

    all_tokens = []
    records = []
    token_offset = 0

    # ------------------------------------------------------------------
    # 1. Extract tokens from every video/window.
    # ------------------------------------------------------------------
    for vi, video_path in enumerate(video_paths):
        video_path = Path(video_path)
        stem = video_path.stem

        info = get_video_info(video_path)
        duration = effective_duration(info, args.max_duration_sec)
        starts = make_window_starts(duration, args.segment_sec, args.stride_sec)

        print(
            f"[joint] video {vi + 1}/{len(video_paths)}: {stem}, "
            f"duration={duration:.3f}s, windows={len(starts)}"
        )

        extraction = extract_window_segments(
            video_path=video_path,
            starts=starts,
            visual_encoder=visual_encoder,
            device=device,
            args=args,
            log_prefix="joint",
            record_prefix={
                "video_index": vi,
                "video_path": str(video_path),
                "video_stem": stem,
                "video_info": info,
            },
            initial_token_offset=token_offset,
        )
        token_offset = extraction.next_token_offset
        all_tokens.append(extraction.tokens_flat)
        records.extend(extraction.records)

    X_all = torch.cat(all_tokens, dim=0)
    print(f"[joint] total token matrix: {tuple(X_all.shape)}")
    torch.save(X_all, out_dir / "joint_tokens_flat.pt")

    # ------------------------------------------------------------------
    # 2. Run ONE NCut over all tokens from all videos.
    # ------------------------------------------------------------------
    print(f"[joint] running one NCut over all videos with num_eig={args.num_eig}")
    eig_all, labels_all, rgb_all = run_embedding_pipeline(
        X_all,
        args=args,
        device=device,
        rgb=True,
    )
    assert rgb_all is not None

    torch.save(eig_all, out_dir / "joint_ncut_eig_flat.pt")

    eig_dims = min(args.eig_rgb_dims, eig_all.shape[1])
    print(f"[joint] computed UMAP RGB from first {eig_dims} eigenvectors")
    np.save(out_dir / "joint_umap_rgb_flat.npy", rgb_all)

    print("[joint] ran shared k-means on joint NCut embedding")
    np.save(out_dir / "joint_cluster_labels_flat.npy", labels_all)

    # ------------------------------------------------------------------
    # 3. Write one output video per input video.
    # ------------------------------------------------------------------
    by_video = group_records_by_key(records, "video_path")

    for video_path_str, video_records in by_video.items():
        video_path = Path(video_path_str)
        stem = video_path.stem
        video_out_dir = out_dir / stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        info = video_records[0]["video_info"]
        video_fps = info["fps"]
        total_frames = info["total_frames"]
        out_size = (args.image_size, args.image_size)

        print(f"[joint] writing overlays for {stem}")

        rgb_mask_accum, cluster_prob_accum, mask_count = accumulate_masks_for_video(
            records=video_records,
            rgb_flat=rgb_all,
            labels_flat=labels_all,
            total_frames=total_frames,
            out_size=out_size,
            num_clusters=args.num_clusters,
        )

        write_held_mask_overlay_videos(
            video_path=video_path,
            rgb_out_path=video_out_dir / "joint_umap_rgb_overlay.mp4",
            cluster_out_path=video_out_dir / "joint_clusters_overlay.mp4",
            rgb_mask_accum=rgb_mask_accum,
            cluster_prob_accum=cluster_prob_accum,
            mask_count=mask_count,
            video_fps=video_fps,
            out_size=out_size,
            max_duration_sec=args.max_duration_sec,
            alpha=args.alpha,
        )

        write_json(
            video_out_dir / "joint_video_segments.json",
            {
                "video": str(video_path),
                "video_info": info,
                "segment_sec": args.segment_sec,
                "stride_sec": args.stride_sec,
                "num_windows": len(video_records),
                "segments": records_without_images(video_records),
            },
        )

        print(f"[joint] wrote {video_out_dir}")

    write_json(
        out_dir / "joint_all_segments.json",
        {
            "video_dir": str(video_dir),
            "num_videos": len(video_paths),
            "num_segments": len(records),
            "num_tokens": int(eig_all.shape[0]),
            "num_eig": int(args.num_eig),
            "eig_rgb_dims": int(args.eig_rgb_dims),
            "segments": records_without_images(records),
        },
    )

    print(f"[joint] all outputs written to: {out_dir}")
