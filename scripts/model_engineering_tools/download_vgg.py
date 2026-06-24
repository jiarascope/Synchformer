import csv
import random
import subprocess
from pathlib import Path

CSV_PATH = Path("data/vggsound.csv")
OUT_DIR = Path("/home/jiaray/mrBean/Synchformer/data/vggsound/h264_video_25fps_256side_16000hz_aac")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 5
random.seed()  # use e.g. random.seed(0) for reproducible samples

with CSV_PATH.open(newline="") as f:
    rows = [r for r in csv.reader(f) if len(r) >= 4]

random.shuffle(rows)

downloaded = 0

for yt_id, start_s, label, split in rows:
    start_s = float(start_s)
    end_s = start_s + 10.0

    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)

    stem = f"{yt_id}_{start_ms}_{end_ms}"
    temp_mp4 = OUT_DIR / f"{stem}.download.mp4"
    final_mp4 = OUT_DIR / f"{stem}.mp4"

    if final_mp4.exists():
        downloaded += 1
        continue

    url = f"https://www.youtube.com/watch?v={yt_id}"

    print(f"Trying: {yt_id} | {start_s:.1f}-{end_s:.1f}s | {label} | {split}")

    try:
        # Download only the requested 10-second section when possible.
        subprocess.run(
            [
                "yt-dlp",
                "--no-warnings",
                "-f", "bv*+ba/b",
                "--download-sections", f"*{start_s}-{end_s}",
                "--force-keyframes-at-cuts",
                "--merge-output-format", "mp4",
                "-o", str(temp_mp4),
                url,
            ],
            check=True,
        )

        # Normalize close to Synchformer-style clips:
        # H.264 video, 25 fps, short side 256, 16 kHz audio.
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(temp_mp4),
                "-t", "10",
                "-vf", "scale='if(gt(iw,ih),256,-2)':'if(gt(iw,ih),-2,256)',fps=25",
                "-ar", "16000",
                "-ac", "1",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(final_mp4),
            ],
            check=True,
        )

        temp_mp4.unlink(missing_ok=True)

        print(f"Saved: {final_mp4}")
        downloaded += 1

        if downloaded >= N:
            break

    except subprocess.CalledProcessError:
        print(f"Failed/unavailable: {yt_id}")
        temp_mp4.unlink(missing_ok=True)
        continue

print(f"Done. Downloaded {downloaded} clips to {OUT_DIR}")