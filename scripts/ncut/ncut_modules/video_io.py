from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def iter_videos(video: Optional[str], video_dir: Optional[str]) -> Iterable[Path]:
    if video:
        yield Path(video)
        return

    if video_dir:
        root = Path(video_dir)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS:
                yield p
        return

    raise ValueError("Pass either --video or --video_dir")

def get_video_info(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        fps = 25.0

    duration = total_frames / fps if total_frames > 0 else 0.0

    return {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "duration_sec": float(duration),
    }
