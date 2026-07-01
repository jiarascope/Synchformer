from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .audio_media import mux_audio
from .audio_spectrogram import make_overlay_image, rasterize_mask
from .audio_tokens import VideoItem


VIDEO_EXTS = {".mp4", ".m4v", ".mov"}


def draw_frame(
    overlay_ft: np.ndarray,
    time_sec: float,
    duration_sec: float,
    width: int,
    height: int,
    margin: int,
    bar_h: int,
    title: str | None = None,
) -> np.ndarray:
    """Make one BGR video frame with cursor and scrollbar."""
    frame = np.full((height, width, 3), 18, dtype=np.uint8)
    title_h = 34 if title else 0
    panel_h = height - 2 * margin - bar_h - 28 - title_h
    panel_w = width - 2 * margin
    panel_h = max(panel_h, 64)
    panel_w = max(panel_w, 64)

    text_y = margin + 22
    y0, x0 = margin + title_h, margin
    if title:
        cv2.putText(frame, title[:150], (x0, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)

    # Flip vertically so low mel bins are lower on screen.
    panel_rgb = cv2.resize(np.flipud(overlay_ft), (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    frame[y0:y0 + panel_h, x0:x0 + panel_w] = panel_rgb[:, :, ::-1]

    rel = 0.0 if duration_sec <= 0 else float(np.clip(time_sec / duration_sec, 0, 1))
    cx = x0 + int(round(rel * (panel_w - 1)))
    cv2.line(frame, (cx, y0), (cx, y0 + panel_h - 1), (255, 255, 255), 2)

    # Border.
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 1)

    # Scroll/progress bar.
    by = y0 + panel_h + 22
    bh = bar_h
    cv2.rectangle(frame, (x0, by), (x0 + panel_w, by + bh), (55, 55, 55), -1)
    cv2.rectangle(frame, (x0, by), (x0 + int(rel * panel_w), by + bh), (210, 210, 210), -1)
    cv2.line(frame, (cx, by - 5), (cx, by + bh + 5), (255, 255, 255), 2)

    label = f"{time_sec:0.2f}s / {duration_sec:0.2f}s"
    cv2.putText(frame, label, (x0, min(height - 12, by + bh + 26)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return frame


def write_video_frames(
    overlay_ft: np.ndarray,
    duration_sec: float,
    temp_video: Path,
    fps: float,
    width: int,
    height: int,
    margin: int,
    scrollbar_height: int,
    title: str | None = None,
) -> None:
    duration = max(float(duration_sec), 1.0 / float(fps))
    # Render enough frames to cover the requested duration. The cursor position
    # is computed from the frame index and the final frame reaches rel=1.0.
    n_frames = max(2, int(math.ceil(duration * fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {temp_video}")
    try:
        denom = max(n_frames - 1, 1)
        for i in tqdm(range(n_frames), desc=f"Writing video {temp_video.name}"):
            # Tie cursor/progress strictly to rendered frame index, not to any
            # feature-window count. This prevents the bar from stopping early.
            t = duration * (float(i) / float(denom))
            frame = draw_frame(overlay_ft, t, duration, width, height, margin, scrollbar_height, title=title)
            writer.write(frame)
    finally:
        writer.release()




def render_item(
    item: VideoItem,
    item_clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    temp_video: Path,
    args,
) -> np.ndarray:
    mask_rgb, coverage = rasterize_mask(
        item.grid.coords,
        item_clusters,
        cluster_rgb,
        tuple(item.fbank.shape),
        desc=f"Rasterizing {item.input_mp4.stem}",
    )
    overlay = make_overlay_image(item.fbank, mask_rgb, coverage, args.alpha)

    title = item.input_mp4.name if args.show_title else None
    print(f"Rendering video frames for {item.input_mp4.name}...")
    write_video_frames(
        overlay_ft=overlay,
        duration_sec=item.duration_sec,
        temp_video=temp_video,
        fps=args.fps,
        width=args.width,
        height=args.height,
        margin=args.margin,
        scrollbar_height=args.scrollbar_height,
        title=title,
    )

    print(f"Muxing original audio track for {item.input_mp4.name}...")
    mux_audio(temp_video, item.input_mp4, item.output_mp4)
    print(f"Wrote: {item.output_mp4}")
    return overlay
