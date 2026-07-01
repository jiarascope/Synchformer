from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import List


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + proc.stdout
            + "\n\nSTDERR:\n"
            + proc.stderr
        )
    return proc


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary not found on PATH: {name}")


def extract_audio_to_wav(mp4_path: Path, wav_path: Path, sample_rate: int) -> None:
    require_binary("ffmpeg")
    run([
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav", str(wav_path),
    ])


def _parse_positive_floats(text: str) -> List[float]:
    vals: List[float] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.upper() == "N/A":
            continue
        try:
            val = float(line)
        except ValueError:
            continue
        if math.isfinite(val) and val > 0:
            vals.append(val)
    return vals


def probe_media_durations_sec(path: Path) -> dict[str, float]:
    """Return duration candidates from ffprobe metadata.

    Some edited/fragmented MP4s report a short container or video duration even
    though the decodable audio is longer. For the spectrogram cursor we do not
    want to trust any single metadata field blindly, so callers should combine
    these candidates with the decoded WAV duration and choose the largest sane
    value.
    """
    require_binary("ffprobe")
    out: dict[str, float] = {}

    queries = [
        ("format", [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]),
        ("audio_stream", [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]),
        ("video_stream", [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]),
    ]

    for name, cmd in queries:
        proc = run(cmd, check=False)
        if proc.returncode == 0:
            vals = _parse_positive_floats(proc.stdout)
            if vals:
                out[name] = max(vals)

    return out


def probe_media_duration_sec(path: Path) -> float:
    """Compatibility wrapper: return the largest ffprobe-reported duration."""
    durations = probe_media_durations_sec(path)
    if durations:
        return max(durations.values())
    raise RuntimeError(f"Could not determine media duration with ffprobe: {path}")


def mux_audio(video_no_audio: Path, source_mp4: Path, out_mp4: Path) -> None:
    """Mux original audio into the generated video. Try stream-copy first, AAC fallback."""
    require_binary("ffmpeg")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd_copy = [
        "ffmpeg", "-y",
        "-i", str(video_no_audio), "-i", str(source_mp4),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy", str(out_mp4),
    ]
    proc = run(cmd_copy, check=False)
    if proc.returncode == 0:
        return
    cmd_aac = [
        "ffmpeg", "-y",
        "-i", str(video_no_audio), "-i", str(source_mp4),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        str(out_mp4),
    ]
    run(cmd_aac)


