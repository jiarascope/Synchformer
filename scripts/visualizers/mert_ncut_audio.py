#!/usr/bin/env python3
"""
MERT all-layer JOINT clustering visualizer for MP4 directories.

This version fixes the main problem with "cluster each layer separately":

  OLD behavior:
    layer 0 has its own clusters/colors
    layer 1 has its own clusters/colors
    ...
    so "red" in layer 0 and "red" in layer 8 are NOT comparable.

  DEFAULT behavior here:
    build one feature matrix containing every selected (video, layer, time-window)
    cluster that matrix ONCE
    render each layer as a row using the SAME cluster labels/colors

This makes layer rows comparable.

Important:
  MERT hidden states are temporal tokens, not spectrogram frequency x time patches.
  The overlay is vertical time bands on the spectrogram.

Usage:

  CUDA_VISIBLE_DEVICES=0 python3 /home/jiaray/mrBean/Synchformer/scripts/visualizers/mert_ncut_audio.py \
    /home/jiaray/mrBean/data/baseline_data/conductingValid_clips \
    /home/jiaray/mrBean/plots/mert_outputs/50ev_validclips \
    --device cuda \
    --model m-a-p/MERT-v1-95M \
    --layers all \
    --mode window \
    --segment-sec 0.64 \
    --hop-sec 0.32 \
    --layer-cluster-mode joint_layers \
    --cluster-backend ncut \
    --n-clusters 50 \
    --n-eig 50

Modes:
  --layer-cluster-mode joint_layers
      One shared clustering over all (layer, window) tokens.
      This is the recommended all-layer visualization.

  --layer-cluster-mode separate_layers
      Old diagnostic behavior: one clustering per layer.
      Colors are not comparable across rows.

  --layer-cluster-mode concat_layers
      For each time window, concatenate selected layers into one giant feature.
      Cluster windows once. Every row will show the same labels because the unit
      being clustered is the whole layer stack, not individual layers.

Dependencies:
  pip install torch transformers librosa scikit-learn umap-learn opencv-python tqdm
  pip install ncut-pytorch   # optional for --cluster-backend ncut
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm"}


# -----------------------------
# System helpers
# -----------------------------

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


def ffmpeg_decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    require_binary("ffmpeg")
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-"
    ]
    raw = subprocess.check_output(cmd)
    wav = np.frombuffer(raw, dtype=np.float32)
    if wav.size == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples from {path}")
    return wav


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


def mux_audio(video_no_audio: Path, source_mp4: Path, out_mp4: Path) -> None:
    require_binary("ffmpeg")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd_copy = [
        "ffmpeg", "-y",
        "-i", str(video_no_audio), "-i", str(source_mp4),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_mp4),
    ]
    proc = run(cmd_copy, check=False)
    if proc.returncode == 0:
        return

    cmd_aac = [
        "ffmpeg", "-y",
        "-i", str(video_no_audio), "-i", str(source_mp4),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out_mp4),
    ]
    run(cmd_aac)


# -----------------------------
# MERT extraction
# -----------------------------

class MERTAllLayerExtractor:
    def __init__(self, model_name: str, device: str, half: bool = False):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.model_name = model_name
        self.device = torch.device(device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.sample_rate = int(self.processor.sampling_rate)

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device).eval()

        self.half = bool(half)
        if self.half:
            self.model.half()

    @torch.no_grad()
    def _forward_chunk_all_layers(self, wav_chunk: np.ndarray) -> List[np.ndarray]:
        inputs = self.processor(
            wav_chunk,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.half and "input_values" in inputs:
            inputs["input_values"] = inputs["input_values"].half()

        out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states

        if isinstance(hs, torch.Tensor):
            if hs.ndim != 4:
                raise RuntimeError(f"Unexpected hidden_states tensor shape: {tuple(hs.shape)}")
            hs_list = [hs[i] for i in range(hs.shape[0])]
        else:
            hs_list = list(hs)

        result = []
        for h in hs_list:
            if h.ndim != 3:
                raise RuntimeError(f"Expected hidden state [B,T,C], got {tuple(h.shape)}")
            result.append(h[0].detach().float().cpu().numpy())

        return result

    @torch.no_grad()
    def extract_all_layers_over_audio(
        self,
        wav: np.ndarray,
        chunk_sec: float,
        chunk_hop_sec: float,
    ) -> Tuple[List[np.ndarray], np.ndarray, dict]:
        sr = self.sample_rate
        n = int(len(wav))
        duration_sec = n / float(sr)

        chunk_samples = max(1, int(round(chunk_sec * sr)))
        hop_samples = max(1, int(round(chunk_hop_sec * sr)))
        if hop_samples > chunk_samples:
            raise ValueError("--chunk-hop-sec cannot exceed --chunk-sec")

        starts = list(range(0, max(n, 1), hop_samples))
        if len(starts) > 1 and starts[-2] + chunk_samples >= n:
            starts = starts[:-1]

        layer_chunks: List[List[np.ndarray]] | None = None
        center_chunks: List[np.ndarray] = []
        first_shapes = None

        for st in tqdm(starts, desc="MERT chunks"):
            en = min(n, st + chunk_samples)
            chunk = wav[st:en]
            if chunk.size == 0:
                continue

            hs_list = self._forward_chunk_all_layers(chunk)
            if layer_chunks is None:
                layer_chunks = [[] for _ in hs_list]
                first_shapes = []
                print(f"MERT returned {len(hs_list)} hidden-state entries.")
                for i, h in enumerate(hs_list):
                    first_shapes.append([int(h.shape[0]), int(h.shape[1])])
                    print(f"  layer {i}: [T={h.shape[0]}, C={h.shape[1]}] on first chunk")

            if len(hs_list) != len(layer_chunks):
                raise RuntimeError("Number of hidden-state entries changed across chunks.")

            chunk_dur = (en - st) / float(sr)
            T = int(hs_list[0].shape[0])
            token_rate = T / max(chunk_dur, 1e-9)
            centers = st / float(sr) + (np.arange(T) + 0.5) / token_rate
            center_chunks.append(centers.astype(np.float64))

            for li, h in enumerate(hs_list):
                if int(h.shape[0]) != T:
                    raise RuntimeError(
                        f"Layer {li} has T={h.shape[0]}, expected {T}; "
                        "this script assumes aligned hidden layers."
                    )
                layer_chunks[li].append(h.astype(np.float32))

        if layer_chunks is None:
            raise RuntimeError("No MERT features extracted.")

        centers = np.concatenate(center_chunks, axis=0)
        order = np.argsort(centers)
        centers = centers[order]

        layers_H = []
        for chunks in layer_chunks:
            H = np.concatenate(chunks, axis=0)
            H = H[order]
            layers_H.append(H.astype(np.float32))

        info = {
            "model_name": self.model_name,
            "sample_rate": sr,
            "duration_sec": duration_sec,
            "num_hidden_entries": len(layers_H),
            "first_chunk_layer_shapes_T_C": first_shapes,
            "num_tokens_per_layer_after_concat": int(layers_H[0].shape[0]),
            "approx_feature_rate_hz": float(layers_H[0].shape[0] / max(duration_sec, 1e-9)),
            "chunk_sec": float(chunk_sec),
            "chunk_hop_sec": float(chunk_hop_sec),
        }
        return layers_H, centers, info


def parse_layers_spec(spec: str, num_layers: int) -> List[int]:
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(num_layers))

    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) not in (2, 3):
                raise ValueError(f"Bad layer range: {part}")
            start = int(bits[0]) if bits[0] else 0
            stop = int(bits[1]) if bits[1] else num_layers
            step = int(bits[2]) if len(bits) == 3 and bits[2] else 1
            layers.extend(list(range(start, stop, step)))
        else:
            layers.append(int(part))

    resolved = []
    for l in layers:
        if l < 0:
            l = num_layers + l
        if l < 0 or l >= num_layers:
            raise ValueError(f"Layer {l} out of range. Available: 0..{num_layers - 1}")
        if l not in resolved:
            resolved.append(l)
    return resolved


def token_intervals_from_centers(centers: np.ndarray, duration_sec: float) -> List[Tuple[float, float]]:
    if len(centers) == 1:
        return [(0.0, float(duration_sec))]
    mids = 0.5 * (centers[:-1] + centers[1:])
    starts = np.concatenate([[0.0], mids])
    ends = np.concatenate([mids, [duration_sec]])
    starts = np.maximum(starts, 0.0)
    ends = np.minimum(ends, duration_sec)
    return [(float(a), float(b)) for a, b in zip(starts, ends)]


def pool_windows(
    H: np.ndarray,
    centers: np.ndarray,
    duration_sec: float,
    segment_sec: float,
    hop_sec: float,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    starts = np.arange(0.0, max(0.0, duration_sec - segment_sec) + 1e-9, hop_sec)
    if starts.size == 0:
        starts = np.asarray([0.0], dtype=np.float64)

    feats = []
    intervals = []
    for st in starts:
        en = min(float(duration_sec), float(st + segment_sec))
        mask = (centers >= st) & (centers < en)
        if np.any(mask):
            feat = H[mask].mean(axis=0)
        else:
            mid = 0.5 * (st + en)
            feat = H[int(np.argmin(np.abs(centers - mid)))]
        feats.append(feat.astype(np.float32))
        intervals.append((float(st), float(en)))

    return np.stack(feats, axis=0), intervals


# -----------------------------
# Spectrogram
# -----------------------------

def compute_mel_db(
    wav: np.ndarray,
    sr: int,
    n_mels: int,
    frame_shift_ms: float,
    n_fft: int,
) -> Tuple[np.ndarray, float]:
    import librosa

    hop_length = max(1, int(round(sr * frame_shift_ms / 1000.0)))
    S = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    frame_hop_sec = hop_length / float(sr)
    return S_db, frame_hop_sec


# -----------------------------
# Data
# -----------------------------

@dataclass
class VideoItem:
    input_mp4: Path
    output_mp4: Path
    spec_db: np.ndarray
    spec_frame_hop_sec: float
    duration_sec: float
    sample_rate: int
    hidden_info: dict
    layer_features: Dict[int, np.ndarray] = field(default_factory=dict)
    layer_intervals: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    layer_offsets: Dict[int, Tuple[int, int]] = field(default_factory=dict)


# -----------------------------
# Clustering
# -----------------------------

def _standardize_pca_l2(features: np.ndarray, pca_dim: int = 64) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(features, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected [N,C] features, got {X.shape}")

    X = StandardScaler().fit_transform(X)
    if X.shape[0] > 2 and X.shape[1] > pca_dim:
        dim = min(pca_dim, X.shape[1], X.shape[0] - 1)
        X = PCA(n_components=dim, random_state=0).fit_transform(X).astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X.astype(np.float32)


def run_ncut_or_cluster(
    features: np.ndarray,
    n_eig: int,
    n_clusters: int,
    device: str,
    seed: int,
    backend: str,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(features.shape[0])
    if n == 0:
        raise ValueError(f"{name}: no features")
    if n_clusters > n:
        print(f"WARNING: {name}: reducing n_clusters {n_clusters} -> {n}")
        n_clusters = n

    Xp = _standardize_pca_l2(features, pca_dim=max(64, n_eig, n_clusters))

    if backend == "ncut":
        try:
            from ncut_pytorch import Ncut
            from sklearn.cluster import MiniBatchKMeans, KMeans

            x = torch.from_numpy(Xp).float().to(device)
            print(f"{name}: running Ncut(n_eig={n_eig}) on {n:,} nodes...")
            eig = Ncut(n_eig=int(n_eig)).fit_transform(x).detach().float().cpu().numpy()
            z = eig[:, :min(int(n_clusters), eig.shape[1])]

            try:
                km = MiniBatchKMeans(
                    n_clusters=int(n_clusters),
                    random_state=seed,
                    n_init=10,
                    batch_size=max(1024, 3 * int(n_clusters)),
                    reassignment_ratio=0.0,
                )
                labels = km.fit_predict(z).astype(np.int32)
            except Exception:
                km = KMeans(n_clusters=int(n_clusters), random_state=seed, n_init=10)
                labels = km.fit_predict(z).astype(np.int32)

            _print_cluster_diagnostics(name, labels, n_clusters)
            return eig.astype(np.float32), labels
        except Exception as exc:
            print(f"{name}: NCut failed ({exc}). Falling back to kmeans.")
            backend = "kmeans"

    if backend == "spectral":
        from sklearn.cluster import SpectralClustering
        n_neighbors = min(12, max(1, n - 1))
        labels = SpectralClustering(
            n_clusters=int(n_clusters),
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=seed
        ).fit_predict(Xp).astype(np.int32)
        _print_cluster_diagnostics(name, labels, n_clusters)
        return Xp, labels

    if backend == "kmeans":
        from sklearn.cluster import MiniBatchKMeans, KMeans
        try:
            km = MiniBatchKMeans(
                n_clusters=int(n_clusters),
                random_state=seed,
                n_init=10,
                batch_size=max(1024, 3 * int(n_clusters)),
                reassignment_ratio=0.0,
            )
            labels = km.fit_predict(Xp).astype(np.int32)
        except Exception:
            km = KMeans(n_clusters=int(n_clusters), random_state=seed, n_init=10)
            labels = km.fit_predict(Xp).astype(np.int32)
        _print_cluster_diagnostics(name, labels, n_clusters)
        return Xp, labels

    raise ValueError(f"Unknown --cluster-backend: {backend}")


def _print_cluster_diagnostics(name: str, clusters: np.ndarray, requested_clusters: int) -> None:
    unique, counts = np.unique(clusters, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = ", ".join(f"{int(unique[j])}:{int(counts[j])}" for j in order[:12])
    print(
        f"{name}: requested_clusters={requested_clusters}, "
        f"actual_unique_clusters={len(unique)}, nodes={len(clusters):,}"
    )
    print(f"{name}: largest cluster sizes: {top}")


# -----------------------------
# Color mapping
# -----------------------------

def _pca3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
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
    return 0.15 + 0.85 * y.astype(np.float32)


def embed_cluster_colors(points: np.ndarray, clusters: np.ndarray, method: str, seed: int) -> np.ndarray:
    n_clusters = int(clusters.max()) + 1
    centroids = []
    for c in range(n_clusters):
        pts = points[clusters == c]
        centroids.append(pts.mean(axis=0) if len(pts) else np.zeros(points.shape[1], dtype=np.float32))
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
            print(f"UMAP color embedding failed ({exc}); falling back to PCA.")
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
    elif method == "pca":
        emb = _pca3(centroids)
    else:
        raise ValueError(method)

    return _normalize_rgb(emb)


# -----------------------------
# Rasterization/rendering
# -----------------------------

def rasterize_time_band_mask(
    intervals: Sequence[Tuple[float, float]],
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    spec_shape_ft: Tuple[int, int],
    frame_hop_sec: float,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    Freq, T = spec_shape_ft
    acc = np.zeros((Freq, T, 3), dtype=np.float32)
    cnt = np.zeros((Freq, T, 1), dtype=np.float32)

    for (st, en), c in tqdm(zip(intervals, clusters), total=len(clusters), desc=desc):
        t0 = int(math.floor(st / frame_hop_sec))
        t1 = int(math.ceil(en / frame_hop_sec))
        t0 = max(0, min(T, t0))
        t1 = max(0, min(T, t1))
        if t1 <= t0:
            continue
        color = cluster_rgb[int(c)]
        acc[:, t0:t1, :] += color
        cnt[:, t0:t1, :] += 1.0

    rgb = acc / np.maximum(cnt, 1.0)
    coverage = (cnt[..., 0] > 0).astype(np.float32)
    return rgb, coverage


def make_overlay_image(spec_db: np.ndarray, mask_rgb: np.ndarray, coverage: np.ndarray, alpha: float) -> np.ndarray:
    lo, hi = np.percentile(spec_db, [1, 99])
    gray = np.clip((spec_db - lo) / max(hi - lo, 1e-6), 0, 1)
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    a = (alpha * coverage[..., None]).astype(np.float32)
    out = (1.0 - a) * base + a * mask_rgb
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def make_layer_overlay(
    item: VideoItem,
    layer: int,
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    mask_rgb, coverage = rasterize_time_band_mask(
        intervals=item.layer_intervals[layer],
        clusters=clusters,
        cluster_rgb=cluster_rgb,
        spec_shape_ft=tuple(item.spec_db.shape),
        frame_hop_sec=item.spec_frame_hop_sec,
        desc=f"Rasterizing {item.input_mp4.stem} layer {layer}",
    )
    return make_overlay_image(item.spec_db, mask_rgb, coverage, args.alpha)


def draw_all_layers_frame(
    overlays_by_layer: Dict[int, np.ndarray],
    layers: Sequence[int],
    time_sec: float,
    duration_sec: float,
    width: int,
    row_height: int,
    label_width: int,
    margin: int,
    title_h: int,
    bar_h: int,
    title: str,
) -> np.ndarray:
    n_layers = len(layers)
    panel_w = width - 2 * margin - label_width
    total_h = title_h + n_layers * row_height + bar_h + 2 * margin + 36
    frame = np.full((total_h, width, 3), 18, dtype=np.uint8)

    x_label = margin
    x_panel = margin + label_width
    y = margin

    cv2.putText(
        frame,
        title[:170],
        (margin, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    y += title_h

    rel = 0.0 if duration_sec <= 0 else float(np.clip(time_sec / duration_sec, 0, 1))
    cx = x_panel + int(round(rel * (panel_w - 1)))

    for layer in layers:
        overlay_ft = overlays_by_layer[layer]
        panel_rgb = cv2.resize(np.flipud(overlay_ft), (panel_w, row_height), interpolation=cv2.INTER_AREA)
        frame[y:y + row_height, x_panel:x_panel + panel_w] = panel_rgb[:, :, ::-1]

        cv2.putText(
            frame,
            f"layer {layer}",
            (x_label, y + max(24, row_height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

        cv2.line(frame, (cx, y), (cx, y + row_height - 1), (255, 255, 255), 2)
        cv2.rectangle(frame, (x_panel, y), (x_panel + panel_w, y + row_height), (220, 220, 220), 1)
        y += row_height

    by = y + 18
    cv2.rectangle(frame, (x_panel, by), (x_panel + panel_w, by + bar_h), (55, 55, 55), -1)
    cv2.rectangle(frame, (x_panel, by), (x_panel + int(rel * panel_w), by + bar_h), (210, 210, 210), -1)
    cv2.line(frame, (cx, by - 5), (cx, by + bar_h + 5), (255, 255, 255), 2)

    label = f"{time_sec:0.2f}s / {duration_sec:0.2f}s"
    cv2.putText(
        frame,
        label,
        (x_panel, min(total_h - 12, by + bar_h + 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return frame


def write_all_layers_video(
    overlays_by_layer: Dict[int, np.ndarray],
    layers: Sequence[int],
    duration_sec: float,
    temp_video: Path,
    fps: float,
    width: int,
    row_height: int,
    label_width: int,
    margin: int,
    title_h: int,
    scrollbar_height: int,
    title: str,
) -> None:
    duration = max(float(duration_sec), 1.0 / float(fps))
    n_frames = max(2, int(math.ceil(duration * fps)))
    height = title_h + len(layers) * row_height + scrollbar_height + 2 * margin + 36

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {temp_video}")
    try:
        denom = max(n_frames - 1, 1)
        for i in tqdm(range(n_frames), desc=f"Writing all-layer video {temp_video.name}"):
            t = duration * (float(i) / float(denom))
            frame = draw_all_layers_frame(
                overlays_by_layer=overlays_by_layer,
                layers=layers,
                time_sec=t,
                duration_sec=duration,
                width=width,
                row_height=row_height,
                label_width=label_width,
                margin=margin,
                title_h=title_h,
                bar_h=scrollbar_height,
                title=title,
            )
            writer.write(frame)
    finally:
        writer.release()


def write_layer_token_csv(item: VideoItem, layer: int, clusters: np.ndarray, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "layer", "cluster", "start_sec", "end_sec", "video_path"])
        for i, ((st, en), c) in enumerate(zip(item.layer_intervals[layer], clusters)):
            w.writerow([i, layer, int(c), float(st), float(en), str(item.input_mp4)])


# -----------------------------
# Input/output and item prep
# -----------------------------

def discover_input_videos(input_path: Path, pattern: str, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Input file does not look like supported video: {input_path}")
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


def output_path_for_video(video_path: Path, input_root: Path, output_base: Path, suffix: str, single_file_mode: bool) -> Path:
    if single_file_mode:
        return output_base
    try:
        rel = video_path.relative_to(input_root)
    except ValueError:
        rel = Path(video_path.name)
    rel_parent = rel.parent if str(rel.parent) != "." else Path("")
    return output_base / rel_parent / f"{video_path.stem}{suffix}.mp4"


def prepare_video_item(
    video_path: Path,
    output_path: Path,
    extractor: MERTAllLayerExtractor,
    selected_layers_spec: str,
    args: argparse.Namespace,
) -> Tuple[VideoItem, List[int]]:
    print(f"\n=== Extracting {video_path.name} ===")
    wav = ffmpeg_decode_audio(video_path, extractor.sample_rate)
    wav_duration_sec = len(wav) / float(extractor.sample_rate)

    media_durations = probe_media_durations_sec(video_path)
    ffprobe_duration_sec = max(media_durations.values()) if media_durations else 0.0
    duration_sec = max(wav_duration_sec, ffprobe_duration_sec)

    print(
        "Duration candidates: "
        + ", ".join([f"ffprobe_{k}={v:.3f}s" for k, v in sorted(media_durations.items())])
        + f", decoded_wav={wav_duration_sec:.3f}s, render={duration_sec:.3f}s"
    )

    print("Extracting all MERT hidden states once...")
    layers_H, centers, info = extractor.extract_all_layers_over_audio(
        wav=wav,
        chunk_sec=args.chunk_sec,
        chunk_hop_sec=args.chunk_hop_sec,
    )
    selected_layers = parse_layers_spec(selected_layers_spec, len(layers_H))
    print(f"Selected layers: {selected_layers}")

    print("Computing display spectrogram...")
    spec_db, spec_frame_hop_sec = compute_mel_db(
        wav=wav,
        sr=extractor.sample_rate,
        n_mels=args.n_mels,
        frame_shift_ms=args.frame_shift_ms,
        n_fft=args.n_fft,
    )

    item = VideoItem(
        input_mp4=video_path,
        output_mp4=output_path,
        spec_db=spec_db,
        spec_frame_hop_sec=float(spec_frame_hop_sec),
        duration_sec=float(duration_sec),
        sample_rate=int(extractor.sample_rate),
        hidden_info=info,
    )

    for layer in selected_layers:
        H = layers_H[layer]
        if args.mode == "token":
            feats = H
            intervals = token_intervals_from_centers(centers, duration_sec)
        else:
            feats, intervals = pool_windows(
                H=H,
                centers=centers,
                duration_sec=duration_sec,
                segment_sec=args.segment_sec,
                hop_sec=args.hop_sec,
            )
        item.layer_features[layer] = feats.astype(np.float32)
        item.layer_intervals[layer] = intervals
        print(f"{video_path.name} layer {layer}: units={feats.shape[0]:,}, dim={feats.shape[1]}")

    return item, selected_layers


def assign_joint_layer_offsets(items: Sequence[VideoItem], layers: Sequence[int]) -> np.ndarray:
    feats = []
    cur = 0
    for item in items:
        for layer in layers:
            f = item.layer_features[layer]
            n = int(f.shape[0])
            item.layer_offsets[layer] = (cur, cur + n)
            cur += n
            feats.append(f)
    return np.concatenate(feats, axis=0).astype(np.float32)


def assign_layer_offsets(items: Sequence[VideoItem], layer: int) -> np.ndarray:
    feats = []
    cur = 0
    for item in items:
        f = item.layer_features[layer]
        n = int(f.shape[0])
        item.layer_offsets[layer] = (cur, cur + n)
        cur += n
        feats.append(f)
    return np.concatenate(feats, axis=0).astype(np.float32)


def build_concat_layer_features(items: Sequence[VideoItem], layers: Sequence[int]) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int]]]:
    """
    Cluster one feature per time window by concatenating all selected layer reps.
    Returns global concat feature matrix and offsets keyed by (item_idx, layer).
    All layers for the same item share the same offsets/labels.
    """
    all_concat = []
    item_offsets = {}
    cur = 0

    for item_idx, item in enumerate(items):
        n0 = item.layer_features[layers[0]].shape[0]
        for layer in layers:
            if item.layer_features[layer].shape[0] != n0:
                raise RuntimeError("concat_layers requires same number of windows/tokens across selected layers.")
        concat = np.concatenate([item.layer_features[layer] for layer in layers], axis=1).astype(np.float32)
        n = int(concat.shape[0])
        for layer in layers:
            item_offsets[(item_idx, layer)] = (cur, cur + n)
        cur += n
        all_concat.append(concat)

    return np.concatenate(all_concat, axis=0).astype(np.float32), item_offsets


# -----------------------------
# CLI/main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="All-layer MERT visualization with JOINT layer clustering by default."
    )
    p.add_argument("input", type=Path, help="Input MP4 file or directory.")
    p.add_argument("output", type=Path, help="Output MP4 for one file, or output directory.")

    p.add_argument("--glob", default="*.mp4")
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--model", default="m-a-p/MERT-v1-95M")
    p.add_argument("--layers", default="all", help="all, 0,2,4,6,8,10,12, or range like 0:13:2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--half", action="store_true")

    p.add_argument("--chunk-sec", type=float, default=20.0)
    p.add_argument("--chunk-hop-sec", type=float, default=20.0)

    p.add_argument("--mode", choices=["window", "token"], default="window")
    p.add_argument("--segment-sec", type=float, default=0.64)
    p.add_argument("--hop-sec", type=float, default=0.32)

    p.add_argument(
        "--layer-cluster-mode",
        choices=["joint_layers", "separate_layers", "concat_layers"],
        default="joint_layers",
        help=(
            "joint_layers: one shared clustering over all (layer,window) tokens. "
            "separate_layers: one clustering per layer. "
            "concat_layers: concatenate layers per window and cluster windows once."
        ),
    )

    p.add_argument("--cluster-backend", choices=["ncut", "kmeans", "spectral"], default="ncut")
    p.add_argument("--n-clusters", "--n_clusters", type=int, default=12)
    p.add_argument("--n-eig", "--n_eig", type=int, default=12)
    p.add_argument("--embedder", choices=["umap", "tsne", "pca"], default="umap")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--frame-shift-ms", type=float, default=10.0)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--alpha", type=float, default=0.45)

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--montage-width", type=int, default=1800)
    p.add_argument("--row-height", type=int, default=115)
    p.add_argument("--label-width", type=int, default=130)
    p.add_argument("--margin", type=int, default=28)
    p.add_argument("--title-height", type=int, default=42)
    p.add_argument("--scrollbar-height", type=int, default=18)

    p.add_argument("--output-suffix", default="_mert_joint_layers")
    p.add_argument("--write-token-csv", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "token":
        print("WARNING: --mode token can produce many nodes. --mode window is closer to 0.64s/50%-overlap.")

    input_path = args.input.expanduser().resolve()
    output_base = args.output.expanduser().resolve()
    videos = discover_input_videos(input_path, args.glob, args.recursive)

    single_file_mode = input_path.is_file() and len(videos) == 1
    if single_file_mode and output_base.suffix.lower() != ".mp4":
        raise ValueError("For a single input file, output should be an .mp4 path.")

    input_root = input_path.parent if input_path.is_file() else input_path
    if single_file_mode:
        output_base.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_base.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(videos)} input video(s).")
    print(f"Loading MERT: {args.model}")
    extractor = MERTAllLayerExtractor(args.model, args.device, half=args.half)
    print(f"MERT processor sample_rate={extractor.sample_rate} Hz")

    items: List[VideoItem] = []
    selected_layers: List[int] | None = None

    for video in videos:
        out_path = output_path_for_video(
            video_path=video,
            input_root=input_root,
            output_base=output_base,
            suffix=args.output_suffix,
            single_file_mode=single_file_mode,
        )
        item, layers_for_item = prepare_video_item(
            video_path=video,
            output_path=out_path,
            extractor=extractor,
            selected_layers_spec=args.layers,
            args=args,
        )
        if selected_layers is None:
            selected_layers = layers_for_item
        elif selected_layers != layers_for_item:
            raise RuntimeError("Selected layer set changed across videos.")
        items.append(item)

    assert selected_layers is not None
    layers = selected_layers

    with tempfile.TemporaryDirectory(prefix="mert_joint_layers_") as td:
        temp_dir = Path(td)

        # labels_by_item_layer[(item_idx, layer)] = labels for that layer's intervals
        labels_by_item_layer: Dict[Tuple[int, int], np.ndarray] = {}
        rgb_by_item_layer: Dict[Tuple[int, int], np.ndarray] = {}
        shared_rgb: np.ndarray | None = None

        if args.layer_cluster_mode == "joint_layers":
            print("\n=== JOINT clustering over every (video, layer, time-window) token ===")
            joint_features = assign_joint_layer_offsets(items, layers)
            points, labels = run_ncut_or_cluster(
                features=joint_features,
                n_eig=args.n_eig,
                n_clusters=args.n_clusters,
                device=args.device,
                seed=args.seed,
                backend=args.cluster_backend,
                name="Joint-layer MERT clustering",
            )
            shared_rgb = embed_cluster_colors(points, labels, args.embedder, args.seed)

            for item_idx, item in enumerate(items):
                for layer in layers:
                    s, e = item.layer_offsets[layer]
                    labels_by_item_layer[(item_idx, layer)] = labels[s:e]

        elif args.layer_cluster_mode == "concat_layers":
            print("\n=== CONCAT-LAYERS clustering over time windows ===")
            concat_features, concat_offsets = build_concat_layer_features(items, layers)
            points, labels = run_ncut_or_cluster(
                features=concat_features,
                n_eig=args.n_eig,
                n_clusters=args.n_clusters,
                device=args.device,
                seed=args.seed,
                backend=args.cluster_backend,
                name="Concat-layer MERT clustering",
            )
            shared_rgb = embed_cluster_colors(points, labels, args.embedder, args.seed)

            for item_idx, item in enumerate(items):
                for layer in layers:
                    s, e = concat_offsets[(item_idx, layer)]
                    labels_by_item_layer[(item_idx, layer)] = labels[s:e]

        elif args.layer_cluster_mode == "separate_layers":
            print("\n=== SEPARATE clustering per layer ===")
            for layer in layers:
                print(f"\n--- Layer {layer} ---")
                layer_features = assign_layer_offsets(items, layer)
                points, labels = run_ncut_or_cluster(
                    features=layer_features,
                    n_eig=args.n_eig,
                    n_clusters=args.n_clusters,
                    device=args.device,
                    seed=args.seed,
                    backend=args.cluster_backend,
                    name=f"Separate layer {layer}",
                )
                layer_rgb = embed_cluster_colors(points, labels, args.embedder, args.seed)
                for item_idx, item in enumerate(items):
                    s, e = item.layer_offsets[layer]
                    labels_by_item_layer[(item_idx, layer)] = labels[s:e]
                    rgb_by_item_layer[(item_idx, layer)] = layer_rgb

        else:
            raise ValueError(args.layer_cluster_mode)

        for item_idx, item in enumerate(items):
            print(f"\n=== Rendering {item.input_mp4.name} ===")
            overlays_by_layer: Dict[int, np.ndarray] = {}

            for layer in layers:
                labels = labels_by_item_layer[(item_idx, layer)]
                rgb = shared_rgb if shared_rgb is not None else rgb_by_item_layer[(item_idx, layer)]
                assert rgb is not None
                overlay = make_layer_overlay(item, layer, labels, rgb, args)
                overlays_by_layer[layer] = overlay

                if args.write_token_csv:
                    csv_path = (
                        item.output_mp4.with_name(item.output_mp4.stem + f"_layer{layer:02d}.csv")
                        if single_file_mode
                        else output_base / "csv" / f"layer_{layer:02d}" / f"{item.input_mp4.stem}.csv"
                    )
                    write_layer_token_csv(item, layer, labels, csv_path)

            temp_video = temp_dir / f"{item_idx:04d}_all_layers_noaudio.mp4"
            title = (
                f"{item.input_mp4.name} | {args.model} | {args.layer_cluster_mode} | "
                f"{args.mode} {args.segment_sec:.2f}s/{args.hop_sec:.2f}s"
            )
            write_all_layers_video(
                overlays_by_layer=overlays_by_layer,
                layers=layers,
                duration_sec=item.duration_sec,
                temp_video=temp_video,
                fps=args.fps,
                width=args.montage_width,
                row_height=args.row_height,
                label_width=args.label_width,
                margin=args.margin,
                title_h=args.title_height,
                scrollbar_height=args.scrollbar_height,
                title=title,
            )
            mux_audio(temp_video, item.input_mp4, item.output_mp4)
            print(f"Wrote montage: {item.output_mp4}")

        summary = {
            "model": args.model,
            "layers": layers,
            "mode": args.mode,
            "segment_sec": args.segment_sec,
            "hop_sec": args.hop_sec,
            "layer_cluster_mode": args.layer_cluster_mode,
            "cluster_backend": args.cluster_backend,
            "n_clusters": args.n_clusters,
            "n_eig": args.n_eig,
            "interpretation": {
                "joint_layers": "Same cluster/color vocabulary is shared across all layer rows.",
                "separate_layers": "Clusters/colors are independent per layer and not comparable across rows.",
                "concat_layers": "One cluster per time window from concatenated layer features; rows repeat the same labels."
            },
            "honest_limitation": (
                "MERT hidden states are temporal tokens. These visualizations show "
                "time-local cluster bands, not frequency-local spectrogram patches."
            ),
            "videos": [
                {
                    "input": str(item.input_mp4),
                    "output_montage": str(item.output_mp4),
                    "duration_sec": item.duration_sec,
                    "mert_info": item.hidden_info,
                    "units_by_layer": {
                        str(layer): int(item.layer_features[layer].shape[0])
                        for layer in layers
                    },
                }
                for item in items
            ],
        }
        summary_path = (
            output_base.with_suffix(".json") if single_file_mode
            else output_base / "mert_joint_layers_summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
