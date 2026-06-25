from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .process_common import (
    accumulate_masks_for_video,
    effective_duration,
    extract_window_segments,
    group_records_by_key,
    make_window_starts,
    run_embedding_pipeline,
    write_held_mask_overlay_video,
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
    out_dir = out_root
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

    # ------------------------------------------------------------------
    # 2. Run ONE NCut over all tokens from all videos.
    # ------------------------------------------------------------------
    print(f"[joint] running one NCut over all videos with num_eig={args.num_eig}")
    eig_all, _labels_all, rgb_all = run_embedding_pipeline(
        X_all,
        args=args,
        device=device,
        rgb=True,
        clusters=False,
    )
    assert rgb_all is not None

    eig_dims = min(args.eig_rgb_dims, eig_all.shape[1])
    print(
        f"[joint] computed {args.embedding_map.upper()} RGB from first "
        f"{eig_dims} eigenvectors"
    )

    # ------------------------------------------------------------------
    # 3. Write one RGB overlay video per input video.
    # ------------------------------------------------------------------
    by_video = group_records_by_key(records, "video_path")

    for video_path_str, video_records in by_video.items():
        video_path = Path(video_path_str)
        stem = video_path.stem

        info = video_records[0]["video_info"]
        video_fps = info["fps"]
        total_frames = info["total_frames"]
        out_size = (args.image_size, args.image_size)

        print(f"[joint] writing overlays for {stem}")

        rgb_mask_accum, mask_count = accumulate_masks_for_video(
            records=video_records,
            rgb_flat=rgb_all,
            total_frames=total_frames,
            out_size=out_size,
        )

        write_held_mask_overlay_video(
            video_path=video_path,
            rgb_out_path=out_dir / f"{stem}_joint_{args.embedding_map}_rgb_overlay.mp4",
            rgb_mask_accum=rgb_mask_accum,
            mask_count=mask_count,
            video_fps=video_fps,
            out_size=out_size,
            max_duration_sec=args.max_duration_sec,
            alpha=args.alpha,
        )

        print(f"[joint] wrote mp4 overlays for {stem} to {out_dir}")

    print(f"[joint] all outputs written to: {out_dir}")
