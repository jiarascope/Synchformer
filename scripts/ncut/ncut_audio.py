#!/usr/bin/env python3
"""
Audio AST-token Nyström NCut visualizer.

Single-file mode:
  Input:  one MP4 with audio
  Output: one MP4 whose video stream is a static spectrogram + semi-transparent
          NCut patch-cluster mask + moving playback cursor/scroll bar, with the
          original MP4 audio track muxed back in.

Directory/joint mode:
  Input:  a directory of MP4s
  Output: one output MP4 per input MP4. AST patch tokens from every input video
          are concatenated into one global feature matrix and clustered with one
          shared NCut solve. The resulting labels/colors are split back to each
          source video for rendering.

Examples:
  # Single MP4, same behavior as the earlier script.
  python audio_ast_ncut_video.py input.mp4 out.mp4 \
      --device cuda --embedder umap --n-clusters 12 --fps 30

  # Joint NCut over every MP4 in a directory.
  python audio_ast_ncut_video.py ./clips ./ncut_outputs \
      --device cuda --embedder umap --n-clusters 12 --n-eig 24

  # Joint NCut recursively, using t-SNE colors and writing a montage PNG.
  python audio_ast_ncut_video.py ./clips ./ncut_outputs \
      --recursive --embedder tsne --write-global-grid

    can 

Notes:
  * Default model is the Hugging Face MIT AudioSet AST checkpoint. Synchformer uses
    an AST audio feature extractor; pass --synchformer-ckpt to best-effort-load
    matching AST audio-tower weights from a Synchformer feature-extractor ckpt.
  * Long audio is split into AST-sized log-mel windows and token patches are
    stitched back to global spectrogram coordinates.
  * Joint mode can create a very large graph. Start with a small directory and
    increase --n-eig/--n-clusters after confirming memory/time behavior.

    run as:

    CUDA_VISIBLE_DEVICES=0 python3 ./scripts/ncut/ncut_audio.py \
    /home/jiaray/mrBean/data/ncut_annotated/ncut_smalltest ./scripts/outputs/audio  \
    --device cuda   \
    --embedder umap  \
    --n-clusters 15 \
    --n-eig 50

"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".m4v", ".mov"}


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


def mux_audio(video_no_audio: Path, source_mp4: Path, out_mp4: Path) -> None:
    """Mux original audio into the generated video. Try stream-copy first, AAC fallback."""
    require_binary("ffmpeg")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd_copy = [
        "ffmpeg", "-y",
        "-i", str(video_no_audio), "-i", str(source_mp4),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(out_mp4),
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
        "-shortest", str(out_mp4),
    ]
    run(cmd_aac)


def kaldi_logmel(
    waveform: torch.Tensor,
    sample_rate: int,
    num_mel_bins: int,
    frame_shift_ms: float,
) -> torch.Tensor:
    """Return unnormalized log-mel/fbank features as (T, F)."""
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=float(sample_rate),
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=frame_shift_ms,
    )


def load_ast(model_name: str, device: str, half: bool = False) -> torch.nn.Module:
    from transformers import ASTForAudioClassification, ASTModel

    # The common MIT checkpoint is published as a classifier. We want the bare
    # transformer so we can use patch-token hidden states, not class logits.
    try:
        clf = ASTForAudioClassification.from_pretrained(model_name)
        model = clf.audio_spectrogram_transformer
    except Exception:
        model = ASTModel.from_pretrained(model_name)

    model.eval().to(device)
    if half:
        model.half()
    return model


def _unwrap_checkpoint(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "module", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def best_effort_load_synchformer_ast_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    """
    Synchformer feature-extractor checkpoints contain both towers. Their exact
    key names can vary by experiment. This tries to load only tensor keys that
    match the target AST model after removing likely audio-tower prefixes.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = _unwrap_checkpoint(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Could not find a state_dict-like mapping in {ckpt_path}")

    target = model.state_dict()
    loaded = {}
    prefixes = [
        "module.",
        "model.",
        "a_encoder.",
        "a_encoder.model.",
        "afeat_extractor.",
        "afeat_extractor.model.",
        "audio_encoder.",
        "audio_encoder.model.",
        "audio_spectrogram_transformer.",
        "model.a_encoder.",
        "model.afeat_extractor.",
        "model.audio_encoder.",
    ]

    def candidate_names(k: str) -> Iterable[str]:
        yield k
        changed = True
        cur = k
        while changed:
            changed = False
            for p in prefixes:
                if cur.startswith(p):
                    cur = cur[len(p):]
                    yield cur
                    changed = True
        # Last-resort suffix match: if a checkpoint key contains a full target key
        # after some project-specific prefix, load it.
        parts = k.split(".")
        for i in range(len(parts)):
            yield ".".join(parts[i:])

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        for cand in candidate_names(k):
            if cand in target and target[cand].shape == v.shape:
                loaded[cand] = v
                break

    if not loaded:
        raise RuntimeError(
            "No AST keys matched the Synchformer checkpoint. The checkpoint may use "
            "a wrapper that does not expose Hugging Face AST names. In that case, "
            "export the audio tower's AST state_dict first, or use --model-name only."
        )

    missing, unexpected = model.load_state_dict(loaded, strict=False)
    print(
        f"Loaded {len(loaded)} AST tensors from {ckpt_path.name}. "
        f"Missing target tensors: {len(missing)}; unexpected: {len(unexpected)}."
    )


@dataclass
class TokenGrid:
    features: torch.Tensor          # (N, D), on CPU initially
    coords: np.ndarray              # (N, 4): f0, f1, t0, t1 in spectrogram-bin/frame coords
    n_freq_patches: int
    n_time_patches_per_window: int


@dataclass
class VideoItem:
    input_mp4: Path
    output_mp4: Path
    wav_path: Path
    fbank: torch.Tensor             # (T, F), unnormalized log-mel/fbank
    waveform_samples: int
    sample_rate: int
    grid: TokenGrid
    token_start: int = 0
    token_end: int = 0

    @property
    def token_count(self) -> int:
        return int(self.grid.features.shape[0])


def infer_patch_grid(config) -> Tuple[int, int]:
    patch_size = int(config.patch_size if isinstance(config.patch_size, int) else config.patch_size[0])
    n_f = (int(config.num_mel_bins) - patch_size) // int(config.frequency_stride) + 1
    n_t = (int(config.max_length) - patch_size) // int(config.time_stride) + 1
    return n_f, n_t


@torch.no_grad()
def extract_ast_patch_tokens(
    model: torch.nn.Module,
    fbank_norm: torch.Tensor,
    device: str,
    batch_windows: int = 4,
    chunk_hop_frames: int | None = None,
    half: bool = False,
    desc: str = "AST windows",
) -> TokenGrid:
    """Extract AST patch tokens over a whole log-mel sequence."""
    cfg = model.config
    max_len = int(cfg.max_length)
    num_mel_bins = int(cfg.num_mel_bins)
    patch_size = int(cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0])
    t_stride = int(cfg.time_stride)
    f_stride = int(cfg.frequency_stride)
    n_f, n_t = infer_patch_grid(cfg)

    if fbank_norm.shape[1] != num_mel_bins:
        raise ValueError(f"Expected {num_mel_bins} mel bins, got {fbank_norm.shape[1]}")

    total_T = int(fbank_norm.shape[0])
    hop = int(chunk_hop_frames or max_len)
    starts = list(range(0, max(total_T, 1), hop))

    all_feats: List[torch.Tensor] = []
    all_coords: List[np.ndarray] = []

    for b0 in tqdm(range(0, len(starts), batch_windows), desc=desc):
        batch_starts = starts[b0:b0 + batch_windows]
        batch_chunks = []
        for s in batch_starts:
            chunk = fbank_norm[s:s + max_len]
            if chunk.shape[0] < max_len:
                pad = torch.zeros(max_len - chunk.shape[0], num_mel_bins, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=0)
            batch_chunks.append(chunk)
        x = torch.stack(batch_chunks, dim=0).to(device)
        if half:
            x = x.half()
        out = model(input_values=x)
        # AST uses class/distillation tokens before patch tokens in the common HF models.
        tok = out.last_hidden_state[:, 2:, :]
        if tok.shape[1] != n_f * n_t:
            # Some variants may only have one special token.
            tok = out.last_hidden_state[:, -n_f * n_t:, :]
        tok = tok.reshape(len(batch_starts), n_f, n_t, tok.shape[-1])

        for bi, s in enumerate(batch_starts):
            feat = tok[bi].reshape(n_f * n_t, -1).float().cpu()
            coords = []
            keep = []
            idx = 0
            for fi in range(n_f):
                f0 = fi * f_stride
                f1 = min(f0 + patch_size, num_mel_bins)
                for ti in range(n_t):
                    t0 = s + ti * t_stride
                    t1 = min(t0 + patch_size, total_T)
                    # Skip padded-only tokens from the final window.
                    if t0 >= total_T or t1 <= 0:
                        idx += 1
                        continue
                    coords.append((f0, f1, t0, t1))
                    keep.append(idx)
                    idx += 1
            all_feats.append(feat[torch.tensor(keep, dtype=torch.long)])
            all_coords.append(np.asarray(coords, dtype=np.int32))

    features = torch.cat(all_feats, dim=0)
    coords = np.concatenate(all_coords, axis=0)
    return TokenGrid(features=features, coords=coords, n_freq_patches=n_f, n_time_patches_per_window=n_t)


def run_ncut(features: torch.Tensor, n_eig: int, n_clusters: int, device: str) -> Tuple[np.ndarray, np.ndarray]:
    from ncut_pytorch import Ncut, kway_ncut

    x = F.normalize(features.to(device), dim=-1)
    eigvecs = Ncut(n_eig=n_eig).fit_transform(x)
    cut_dims = min(n_clusters, eigvecs.shape[1])
    try:
        kway = kway_ncut(eigvecs[:, :cut_dims], n_clusters=n_clusters)
    except TypeError:
        kway = kway_ncut(eigvecs[:, :cut_dims])
    clusters = kway.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
    eig_np = eigvecs.detach().float().cpu().numpy()
    return eig_np, clusters


def embed_cluster_colors(eigvecs: np.ndarray, clusters: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Return RGB float colors in [0, 1], one per cluster id."""
    n_clusters = int(clusters.max()) + 1
    centroids = []
    for c in range(n_clusters):
        pts = eigvecs[clusters == c]
        centroids.append(pts.mean(axis=0) if len(pts) else np.zeros(eigvecs.shape[1], dtype=np.float32))
    centroids = np.asarray(centroids, dtype=np.float32)

    if n_clusters == 1:
        return np.asarray([[1.0, 0.2, 0.2]], dtype=np.float32)

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=max(2, min(15, n_clusters - 1)),
                min_dist=0.05,
                metric="euclidean",
                random_state=seed,
            )
            emb = reducer.fit_transform(centroids)
        except Exception as exc:
            print(f"UMAP failed ({exc}); falling back to PCA.")
            emb = _pca3(centroids)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = max(1, min(30, n_clusters - 1))
        emb = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(centroids)
    else:
        raise ValueError(method)

    return _normalize_rgb(emb)


def _pca3(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    emb = u[:, :min(3, u.shape[1])] * s[:min(3, len(s))]
    if emb.shape[1] < 3:
        emb = np.pad(emb, ((0, 0), (0, 3 - emb.shape[1])))
    return emb


def _normalize_rgb(x: np.ndarray) -> np.ndarray:
    lo = np.percentile(x, 2, axis=0, keepdims=True)
    hi = np.percentile(x, 98, axis=0, keepdims=True)
    y = (x - lo) / np.maximum(hi - lo, 1e-6)
    y = np.clip(y, 0, 1)
    # Avoid very dark colors.
    return 0.15 + 0.85 * y.astype(np.float32)


def rasterize_mask(
    coords: np.ndarray,
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    spec_shape_tf: Tuple[int, int],
    desc: str = "Rasterizing mask",
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize patch colors to spectrogram pixels. Returns rgb(F,T,3), coverage(F,T)."""
    T, Freq = spec_shape_tf
    acc = np.zeros((Freq, T, 3), dtype=np.float32)
    cnt = np.zeros((Freq, T, 1), dtype=np.float32)
    for (f0, f1, t0, t1), c in tqdm(zip(coords, clusters), total=len(clusters), desc=desc):
        color = cluster_rgb[int(c)]
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
    wav_samples: int,
    sample_rate: int,
    temp_video: Path,
    fps: float,
    width: int,
    height: int,
    margin: int,
    scrollbar_height: int,
    title: str | None = None,
) -> None:
    duration = wav_samples / float(sample_rate)
    n_frames = max(1, int(math.ceil(duration * fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {temp_video}")
    try:
        for i in tqdm(range(n_frames), desc=f"Writing video {temp_video.name}"):
            t = min(i / fps, duration)
            frame = draw_frame(overlay_ft, t, duration, width, height, margin, scrollbar_height, title=title)
            writer.write(frame)
    finally:
        writer.release()


def discover_input_videos(input_path: Path, pattern: str, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Input file does not look like a supported video: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if recursive:
        paths = sorted(p for p in input_path.rglob(pattern) if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    else:
        paths = sorted(p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not paths:
        raise FileNotFoundError(f"No videos matching {pattern!r} found in {input_path}")
    return paths


def output_path_for_video(
    video_path: Path,
    input_root: Path,
    output_base: Path,
    suffix: str,
    single_file_mode: bool,
) -> Path:
    if single_file_mode:
        return output_base
    try:
        rel = video_path.relative_to(input_root)
    except ValueError:
        rel = Path(video_path.name)
    rel_parent = rel.parent if str(rel.parent) != "." else Path("")
    return output_base / rel_parent / f"{video_path.stem}{suffix}.mp4"


def load_model_from_args(args: argparse.Namespace) -> torch.nn.Module:
    print("Loading AST...")
    model = load_ast(args.model_name, args.device, half=args.half)
    if int(model.config.num_mel_bins) != args.num_mel_bins:
        raise RuntimeError(
            f"Model expects {model.config.num_mel_bins} mel bins, but --num-mel-bins={args.num_mel_bins}"
        )
    if args.synchformer_ckpt is not None:
        best_effort_load_synchformer_ast_weights(model, args.synchformer_ckpt.expanduser())
        model.to(args.device).eval()
        if args.half:
            model.half()
    return model


def prepare_video_item(
    video_path: Path,
    output_path: Path,
    wav_path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> VideoItem:
    print(f"\n=== Extracting {video_path.name} ===")
    extract_audio_to_wav(video_path, wav_path, args.sample_rate)
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != args.sample_rate:
        raise RuntimeError(f"Expected {args.sample_rate} Hz WAV, got {sr} Hz for {video_path}")

    print("Computing log-mel spectrogram...")
    fbank = kaldi_logmel(waveform, args.sample_rate, args.num_mel_bins, args.frame_shift_ms)
    fbank_norm = (fbank - args.ast_mean) / args.ast_std

    print("Extracting AST patch tokens...")
    grid = extract_ast_patch_tokens(
        model=model,
        fbank_norm=fbank_norm,
        device=args.device,
        batch_windows=args.batch_windows,
        chunk_hop_frames=args.chunk_hop_frames,
        half=args.half,
        desc=f"AST windows {video_path.stem}",
    )
    print(f"{video_path.name}: {grid.features.shape[0]:,} token nodes; dim={grid.features.shape[1]}")

    return VideoItem(
        input_mp4=video_path,
        output_mp4=output_path,
        wav_path=wav_path,
        fbank=fbank,
        waveform_samples=int(waveform.shape[-1]),
        sample_rate=args.sample_rate,
        grid=grid,
    )


def assign_global_offsets(items: Sequence[VideoItem]) -> torch.Tensor:
    starts = []
    ends = []
    cur = 0
    feats = []
    for item in items:
        starts.append(cur)
        cur += item.token_count
        ends.append(cur)
        feats.append(item.grid.features)
    for item, start, end in zip(items, starts, ends):
        item.token_start = start
        item.token_end = end
    return torch.cat(feats, dim=0)


def render_item(
    item: VideoItem,
    item_clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    temp_video: Path,
    args: argparse.Namespace,
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
        wav_samples=item.waveform_samples,
        sample_rate=item.sample_rate,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "NCut on AST audio patch tokens. Accepts either one MP4 or a directory "
            "of MP4s. Directory mode runs one joint/global NCut over all token patches."
        )
    )
    p.add_argument("input", type=Path, help="Input MP4 or directory of MP4s.")
    p.add_argument("output", type=Path, help="Output MP4 in file mode, or output directory in directory mode.")
    p.add_argument("--glob", default="*.mp4", help="Directory-mode glob pattern. Default: *.mp4")
    p.add_argument("--recursive", action="store_true", help="Directory-mode recursive search.")
    p.add_argument("--output-suffix", default="_joint_ncut", help="Directory-mode suffix for each rendered MP4.")
    p.add_argument("--limit-videos", type=int, default=None, help="Optional debug limit on number of input videos.")
    p.add_argument("--write-global-grid", action="store_true", help="Write joint_grid_overlay.png montage in directory mode.")
    p.add_argument("--global-grid-name", default="joint_grid_overlay.png")

    p.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--synchformer-ckpt", type=Path, default=None,
                   help="Optional Synchformer segment feature-extractor checkpoint. Best-effort loads matching AST audio-tower weights.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--half", action="store_true", help="Run AST in float16. Use only on CUDA/compatible GPUs.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--num-mel-bins", type=int, default=128)
    p.add_argument("--frame-shift-ms", type=float, default=10.0)
    p.add_argument("--ast-mean", type=float, default=-4.2677393)
    p.add_argument("--ast-std", type=float, default=4.5689974)
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--chunk-hop-frames", type=int, default=None,
                   help="Hop between AST windows in mel frames. Default: AST max_length, no overlap.")
    p.add_argument("--n-eig", type=int, default=24)
    p.add_argument("--n-clusters", type=int, default=12)
    p.add_argument("--embedder", choices=["umap", "tsne"], default="umap")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask opacity over spectrogram.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--margin", type=int, default=48)
    p.add_argument("--scrollbar-height", type=int, default=22)
    p.add_argument("--show-title", action="store_true", help="Draw input filename at the top of each rendered video.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.half and not str(args.device).startswith("cuda"):
        print("--half requested on a non-CUDA device; disabling half precision.")
        args.half = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    single_file_mode = input_path.is_file()

    videos = discover_input_videos(input_path, args.glob, args.recursive)
    if args.limit_videos is not None:
        videos = videos[:args.limit_videos]
    if not videos:
        raise RuntimeError("No videos selected.")

    if single_file_mode:
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("Single-file mode expects output to be an .mp4 path.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_root = input_path.parent
    else:
        if output_path.suffix.lower() in VIDEO_EXTS:
            raise ValueError("Directory mode expects output to be a directory, not an MP4 file path.")
        output_path.mkdir(parents=True, exist_ok=True)
        input_root = input_path

    print(f"Selected {len(videos)} video(s).")
    if len(videos) > 1:
        print("Joint mode: all AST patch tokens from all videos will be concatenated before NCut.")

    model = load_model_from_args(args)

    with tempfile.TemporaryDirectory(prefix="audio_ast_joint_ncut_") as td:
        tmp = Path(td)
        items: List[VideoItem] = []

        for i, video_path in enumerate(videos):
            out_mp4 = output_path_for_video(
                video_path=video_path,
                input_root=input_root,
                output_base=output_path,
                suffix=args.output_suffix,
                single_file_mode=single_file_mode,
            )
            wav_path = tmp / f"audio_{i:05d}_16k_mono.wav"
            items.append(prepare_video_item(video_path, out_mp4, wav_path, model, args))

        print("\nBuilding one global token feature matrix...")
        global_features = assign_global_offsets(items)
        total_nodes = int(global_features.shape[0])
        print(f"Global token nodes: {total_nodes:,}; feature dim: {global_features.shape[1]}")
        if total_nodes > 200_000:
            print(
                "WARNING: This is a large joint NCut problem. If it runs out of memory, "
                "try fewer/shorter videos, a larger --chunk-hop-frames, or a smaller --n-eig."
            )

        print("Running one global Nyström NCut...")
        eigvecs, clusters = run_ncut(global_features, args.n_eig, args.n_clusters, args.device)

        print(f"Coloring global clusters with {args.embedder.upper()}...")
        cluster_rgb = embed_cluster_colors(eigvecs, clusters, args.embedder, args.seed)

        overlays_for_grid: List[Tuple[Path, np.ndarray]] = []
        for i, item in enumerate(items):
            print(f"\n=== Rendering {item.input_mp4.name} from global NCut labels ===")
            item_clusters = clusters[item.token_start:item.token_end]
            if len(item_clusters) != item.token_count:
                raise RuntimeError(f"Internal split error for {item.input_mp4}")
            temp_video = tmp / f"video_{i:05d}_no_audio.mp4"
            overlay = render_item(item, item_clusters, cluster_rgb, temp_video, args)
            if args.write_global_grid:
                overlays_for_grid.append((item.input_mp4, overlay))

        if args.write_global_grid:
            grid_out = output_path.parent / args.global_grid_name if single_file_mode else output_path / args.global_grid_name
            write_global_grid_image(overlays_for_grid, grid_out)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
