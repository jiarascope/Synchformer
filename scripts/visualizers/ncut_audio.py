#!/usr/bin/env python3
"""
Audio Synchformer/AST-token Nyström NCut visualizer.

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

Notes:
  * Default encoder is Synchformer's audio tower loaded from an AVCLIP
    feature-extractor checkpoint. The checkpoint contains both audio and visual
    towers; this script uses the repo's own AST wrapper to filter the audio keys.
  * Pass --encoder hf_ast to use the previous Hugging Face AST fallback.
  * Long audio is split into AST-sized log-mel windows and token patches are
    stitched back to global spectrogram coordinates.
  * Joint mode can create a very large graph. Start with a small directory and
    increase --n-eig/--n-clusters after confirming memory/time behavior.

CUDA_VISIBLE_DEVICES=0 python3 ./scripts/visualizers/ncut_audio.py \
  /home/jiaray/mrBean/data/baseline_data/-XUgwM_clips \
  /home/jiaray/mrBean/plots/baseline/single_vid \
  --device cuda \
  --encoder avclip \
  --avclip-ckpt /home/jiaray/mrBean/Synchformer/checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
  --embedder umap \
  --n-clusters 30 \
  --n-eig 50
  --no-feature-plot-csv

"""

from __future__ import annotations

import argparse
import csv
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
    try:
        from transformers import ASTForAudioClassification, ASTModel
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "This script needs Hugging Face transformers to instantiate the AST audio encoder.\n"
            "Synchformer's official conda_env.yml includes transformers=4.27.4. Install it in the\n"
            "same environment you use to run ncut_audio.py, for example:\n\n"
            "  python3 -m pip install 'transformers>=4.27,<5'\n\n"
            "or, if you are using conda/mamba:\n\n"
            "  conda install -c huggingface 'transformers=4.27.4'\n\n"
            "Then verify with:\n\n"
            "  python3 -c \"import transformers; print(transformers.__version__)\""
        ) from e

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


def torch_load_trusted_checkpoint(path, map_location="cpu"):
    """Load a local/trusted training checkpoint under PyTorch 2.6+.

    PyTorch 2.6 changed torch.load's default to weights_only=True. Older
    Synchformer AVCLIP checkpoints can contain numpy scalars or other small
    metadata objects, so loading them may require weights_only=False. Only use
    this for checkpoints you trust, such as the official/local Synchformer
    checkpoint you explicitly pass with --avclip-ckpt.
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have the weights_only argument.
        return torch.load(str(path), map_location=map_location)


class TrustedTorchLoadContext:
    """Temporarily make repo-internal torch.load calls load trusted checkpoints.

    Synchformer's AST wrapper calls torch.load(ckpt_path, map_location='cpu')
    internally, so we cannot pass weights_only=False directly. This context
    patches torch.load only around model construction and only fills in
    weights_only=False when the caller did not specify it.
    """
    def __enter__(self):
        self._orig = torch.load

        def _patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return self._orig(*args, **kwargs)

        torch.load = _patched_load
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.load = self._orig
        return False


def best_effort_load_synchformer_ast_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    """
    Synchformer feature-extractor checkpoints contain both towers. Their exact
    key names can vary by experiment. This tries to load only tensor keys that
    match the target AST model after removing likely audio-tower prefixes.
    """
    ckpt = torch_load_trusted_checkpoint(ckpt_path, map_location="cpu")
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




def detect_synchformer_repo_root(repo_root: Path | None = None) -> Path:
    """Find the Synchformer repository root so imports work from scripts/ncut."""
    candidates: List[Path] = []
    if repo_root is not None:
        candidates.append(repo_root.expanduser().resolve())
    candidates.append(Path.cwd().resolve())
    try:
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    except NameError:
        pass

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / "model/modules/feat_extractors/audio/ast.py").exists():
            return cand
    raise RuntimeError(
        "Could not find the Synchformer repo root. Run this script from inside the "
        "Synchformer checkout or pass --repo-root /path/to/Synchformer."
    )


def _checkpoint_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch_load_trusted_checkpoint(ckpt_path, map_location="cpu")
    state = _unwrap_checkpoint(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint does not contain a state_dict-like mapping: {ckpt_path}")
    return state


def checkpoint_looks_like_avclip(ckpt_path: Path) -> bool:
    try:
        state = _checkpoint_state_dict(ckpt_path)
    except Exception:
        return False
    keys = list(state.keys())
    has_audio = any(k.startswith(("module.a_encoder.", "a_encoder.")) for k in keys)
    has_visual = any(k.startswith(("module.v_encoder.", "v_encoder.")) for k in keys)
    return has_audio and has_visual


def find_avclip_checkpoint(repo_root: Path) -> Path:
    """Best-effort search for a segment-level AVCLIP feature-extractor checkpoint."""
    patterns = [
        "logs/avclip_models/**/checkpoints/*.pt",
        "logs/avclip_models/**/*.pt",
        "**/*avclip*.pt",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(repo_root.glob(pat))
    candidates = sorted(set(p for p in candidates if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if checkpoint_looks_like_avclip(cand):
            return cand
    raise RuntimeError(
        "Could not auto-find an AVCLIP feature-extractor checkpoint under the Synchformer repo. "
        "Pass it explicitly with --avclip-ckpt /path/to/epoch_best.pt. The checkpoint should "
        "contain keys like a_encoder.* and v_encoder.*."
    )


def load_synchformer_audio_from_avclip(
    ckpt_path: Path | None,
    repo_root: Path | None,
    device: str,
    half: bool,
    max_spec_t: int,
) -> torch.nn.Module:
    """Load Synchformer's audio AST tower from an AVCLIP checkpoint.

    Synchformer's own AST class handles `.pt` AVCLIP checkpoints by filtering
    `a_encoder.*` keys from the combined audio/visual state_dict. We instantiate
    it in patch-token mode: no frequency/time aggregation, no classifier head,
    and no global segment pooling. The output tokens remain AST spectrogram patch
    descriptors for NCut.
    """
    root = detect_synchformer_repo_root(repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if ckpt_path is None:
        ckpt_path = find_avclip_checkpoint(root)
    ckpt_path = ckpt_path.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"AVCLIP checkpoint not found: {ckpt_path}")

    try:
        from model.modules.feat_extractors.audio.ast import AST as SynchformerAST
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Could not import Synchformer's audio AST wrapper. Make sure you are running "
            "inside the Synchformer environment and pass --repo-root if needed."
        ) from e

    print(f"Loading Synchformer audio tower from AVCLIP checkpoint:\n  {ckpt_path}")
    print(
        "Note: loading this local AVCLIP checkpoint with weights_only=False "
        "for PyTorch 2.6+ compatibility. Only use checkpoints you trust."
    )
    with TrustedTorchLoadContext():
        model = SynchformerAST(
            extract_features=True,
            ckpt_path=str(ckpt_path),
            feat_type="last_hidden_state_no_AUX",
            max_spec_t=int(max_spec_t),
            factorize_freq_time=False,
            agg_freq_module="AveragePooling",
            agg_time_module="Identity",
            add_global_repr=False,
            agg_segments_module="AveragePooling",
            max_segments=1,
        )
    model.eval().to(device)
    if half:
        model.half()
    # Store this for dispatch/diagnostics.
    model._ncut_encoder_kind = "synchformer_avclip_audio"
    model._ncut_avclip_ckpt = str(ckpt_path)
    return model


def get_patch_grid_from_synchformer_ast(model: torch.nn.Module) -> Tuple[int, int]:
    if hasattr(model, "ast") and hasattr(model.ast, "embeddings") and hasattr(model.ast.embeddings, "get_shape"):
        f, t = model.ast.embeddings.get_shape(model.config)
        return int(f), int(t)
    # Fallback to Hugging Face-style config fields.
    return infer_patch_grid(model.config)


@torch.no_grad()
def extract_synchformer_avclip_patch_tokens(
    model: torch.nn.Module,
    fbank_norm: torch.Tensor,
    device: str,
    batch_windows: int = 16,
    chunk_hop_frames: int | None = None,
    half: bool = False,
    desc: str = "Synchformer AVCLIP audio windows",
) -> TokenGrid:
    """Extract audio AST patch tokens using Synchformer's AVCLIP-loaded audio tower.

    Input chunks are shaped like Synchformer audio segments: (B, S, T, F), with
    S=1 for each sliding spectrogram segment. The model is instantiated to return
    raw patch tokens (no CLS/distill tokens), so every returned token maps back to
    a rectangular time-frequency patch.
    """
    cfg = model.config
    max_len = int(getattr(model, "max_spec_t", None) or getattr(cfg, "max_length"))
    num_mel_bins = int(getattr(cfg, "num_mel_bins"))
    patch_size = int(cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0])
    t_stride = int(cfg.time_stride)
    f_stride = int(cfg.frequency_stride)
    n_f, n_t = get_patch_grid_from_synchformer_ast(model)

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

        # Synchformer AST expects (B, S, T, F). Use S=1 sliding audio segment.
        x = torch.stack(batch_chunks, dim=0).unsqueeze(1).to(device)
        if half:
            x = x.half()
        local, _global = model(x, for_loop=False)
        # Expected shape: (B, 1, n_f*n_t, D) from feat_type=last_hidden_state_no_AUX.
        if local.ndim == 4:
            tok = local[:, 0]
        elif local.ndim == 3:
            # Tolerate wrappers returning (B, P, D).
            tok = local
        else:
            raise RuntimeError(
                f"Unexpected Synchformer audio output shape {tuple(local.shape)}. "
                "Expected patch-token output; check that factorize_freq_time=False."
            )
        if tok.shape[1] != n_f * n_t:
            # If special tokens slipped through, keep the last patch grid.
            if tok.shape[1] >= n_f * n_t:
                tok = tok[:, -n_f * n_t:, :]
            else:
                raise RuntimeError(
                    f"Synchformer audio tower returned {tok.shape[1]} tokens, but expected {n_f*n_t} "
                    f"from patch grid {n_f}x{n_t}."
                )
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


def extract_audio_patch_tokens_dispatch(
    model: torch.nn.Module,
    fbank_norm: torch.Tensor,
    args: argparse.Namespace,
    desc: str,
) -> TokenGrid:
    if getattr(model, "_ncut_encoder_kind", "") == "synchformer_avclip_audio":
        return extract_synchformer_avclip_patch_tokens(
            model=model,
            fbank_norm=fbank_norm,
            device=args.device,
            batch_windows=args.batch_windows,
            chunk_hop_frames=args.chunk_hop_frames,
            half=args.half,
            desc=desc.replace("AST", "AVCLIP audio"),
        )
    return extract_ast_patch_tokens(
        model=model,
        fbank_norm=fbank_norm,
        device=args.device,
        batch_windows=args.batch_windows,
        chunk_hop_frames=args.chunk_hop_frames,
        half=args.half,
        desc=desc,
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
    duration_sec: float
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



def _stratified_sample_indices(
    clusters: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Return a deterministic cluster-stratified sample of token indices."""
    n = int(len(clusters))
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    per_cluster: List[np.ndarray] = []
    # First allocate approximately proportional samples per cluster while
    # guaranteeing at least one point for every non-empty cluster.
    remaining = int(max_points)
    for c in unique:
        idx = np.flatnonzero(clusters == c)
        if len(idx) == 0:
            continue
        target = max(1, int(round(max_points * (len(idx) / n))))
        target = min(target, len(idx), remaining)
        if target <= 0:
            continue
        per_cluster.append(rng.choice(idx, size=target, replace=False))
        remaining -= target
        if remaining <= 0:
            break

    sample = np.concatenate(per_cluster) if per_cluster else np.arange(min(max_points, n), dtype=np.int64)
    if len(sample) < max_points:
        chosen = np.zeros(n, dtype=bool)
        chosen[sample] = True
        rest = np.flatnonzero(~chosen)
        extra = rng.choice(rest, size=min(max_points - len(sample), len(rest)), replace=False)
        sample = np.concatenate([sample, extra])
    elif len(sample) > max_points:
        sample = rng.choice(sample, size=max_points, replace=False)

    sample = np.asarray(sample, dtype=np.int64)
    sample.sort()
    return sample


def _embed_points_2d(
    points: np.ndarray,
    method: str,
    seed: int,
    metric: str,
) -> np.ndarray:
    """Embed feature/eigenvector points to 2D for plotting."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected 2D points array, got shape {points.shape}")
    n = int(points.shape[0])
    if n == 0:
        raise ValueError("Cannot plot an empty feature set.")
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)

    if method == "umap":
        try:
            import umap
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "--embedder umap needs umap-learn for the 2D feature scatter plot. Install it with:\n"
                "  python3 -m pip install umap-learn"
            ) from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(30, n - 1)),
            min_dist=0.05,
            metric=metric,
            random_state=seed,
        )
        xy = reducer.fit_transform(points)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        # t-SNE is slow and noisy in very high dimensions. A small PCA pre-step
        # keeps the plot stable without changing cluster colors or labels.
        x = points
        if x.shape[1] > 50 and n > 50:
            x0 = x - x.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(x0, full_matrices=False)
            x = x0 @ vt[:50].T
        perplexity = max(2, min(30, (n - 1) // 3))
        xy = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca" if n > 3 else "random",
            learning_rate="auto",
            metric=metric,
            random_state=seed,
        ).fit_transform(x)
    else:
        raise ValueError(method)

    xy = np.asarray(xy, dtype=np.float32)
    return xy


def _build_global_token_metadata(items: Sequence[VideoItem]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global per-token video ids, local token ids, and spectrogram coords."""
    total = sum(int(item.token_count) for item in items)
    video_ids = np.empty(total, dtype=np.int32)
    local_token_ids = np.empty(total, dtype=np.int64)
    coords = np.empty((total, 4), dtype=np.int32)

    for vi, item in enumerate(items):
        s = int(item.token_start)
        e = int(item.token_end)
        n = e - s
        if n != item.token_count:
            raise RuntimeError(
                f"Internal token offset mismatch for {item.input_mp4}: "
                f"offsets give {n}, grid has {item.token_count}"
            )
        video_ids[s:e] = vi
        local_token_ids[s:e] = np.arange(n, dtype=np.int64)
        coords[s:e] = item.grid.coords.astype(np.int32, copy=False)
    return video_ids, local_token_ids, coords


def _resolve_feature_plot_csv_path(out_png: Path, csv_name: str | None, output_base: Path) -> Path:
    if csv_name is None:
        return out_png.with_name(out_png.stem + "_points.csv")
    out_csv = Path(csv_name).expanduser()
    if not out_csv.is_absolute():
        out_csv = output_base / out_csv
    return out_csv


def _write_feature_plot_point_csv(
    out_csv: Path,
    sample_idx: np.ndarray,
    xy: np.ndarray,
    sampled_clusters: np.ndarray,
    items: Sequence[VideoItem],
    video_ids: np.ndarray,
    local_token_ids: np.ndarray,
    coords: np.ndarray,
    frame_shift_ms: float,
) -> None:
    """Write exact metadata for every point that appears in the feature plot."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_to_sec = float(frame_shift_ms) / 1000.0
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "plot_row",
            "global_token_index",
            "x",
            "y",
            "cluster",
            "video_index",
            "video_name",
            "video_path",
            "local_token_index",
            "freq_bin_start",
            "freq_bin_end",
            "time_frame_start",
            "time_frame_end",
            "time_sec_start",
            "time_sec_end",
        ])
        for row, global_idx in enumerate(sample_idx):
            vi = int(video_ids[global_idx])
            item = items[vi]
            f0, f1, t0, t1 = [int(v) for v in coords[global_idx]]
            writer.writerow([
                row,
                int(global_idx),
                float(xy[row, 0]),
                float(xy[row, 1]),
                int(sampled_clusters[row]),
                vi,
                item.input_mp4.name,
                str(item.input_mp4),
                int(local_token_ids[global_idx]),
                f0,
                f1,
                t0,
                t1,
                float(t0 * frame_to_sec),
                float(t1 * frame_to_sec),
            ])
    print(f"Wrote feature-plot point metadata CSV: {out_csv}")


def write_feature_embedding_plot(
    points: np.ndarray,
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    out_png: Path,
    method: str,
    seed: int,
    max_points: int,
    source_name: str,
    metric: str,
    items: Sequence[VideoItem],
    frame_shift_ms: float,
    dpi: int = 180,
    video_markers: bool = True,
    video_labels: bool = True,
    video_legend_limit: int = 20,
    out_csv: Path | None = None,
    write_csv: bool = True,
) -> None:
    """Write a 2D UMAP/t-SNE scatter plot of global token points.

    Each plotted point is one audio patch token. Point face colors are exactly
    the same cluster RGB values used for the spectrogram mask. Source video is
    encoded separately with marker shape and optional video-centroid labels, so
    the plot can show both cluster identity and video provenance.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Writing the feature embedding plot needs matplotlib. Install it with:\n"
            "  python3 -m pip install matplotlib"
        ) from exc

    clusters = np.asarray(clusters, dtype=np.int32)
    if len(items) == 0:
        raise ValueError("Cannot write video-aware feature plot with zero video items.")

    video_ids, local_token_ids, token_coords = _build_global_token_metadata(items)
    if len(video_ids) != len(clusters):
        raise RuntimeError(
            f"Feature plot metadata has {len(video_ids)} tokens, but clusters has {len(clusters)} labels."
        )

    sample_idx = _stratified_sample_indices(clusters, max_points=max_points, seed=seed)
    sampled_points = np.asarray(points[sample_idx], dtype=np.float32)
    sampled_clusters = clusters[sample_idx]
    sampled_video_ids = video_ids[sample_idx]

    print(
        f"Embedding {len(sample_idx):,}/{len(clusters):,} global token points with "
        f"{method.upper()} for feature scatter plot..."
    )
    xy = _embed_points_2d(sampled_points, method=method, seed=seed, metric=metric)
    colors = cluster_rgb[sampled_clusters]

    if write_csv:
        if out_csv is None:
            out_csv = out_png.with_name(out_png.stem + "_points.csv")
        _write_feature_plot_point_csv(
            out_csv=out_csv,
            sample_idx=sample_idx,
            xy=xy,
            sampled_clusters=sampled_clusters,
            items=items,
            video_ids=video_ids,
            local_token_ids=local_token_ids,
            coords=token_coords,
            frame_shift_ms=frame_shift_ms,
        )

    x = xy[:, 0]
    y = xy[:, 1]
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=dpi)

    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "H", "p", "8"]
    if video_markers:
        # Cluster identity remains the face color. Video identity is marker shape.
        for vi in np.unique(sampled_video_ids):
            m = sampled_video_ids == vi
            marker = marker_cycle[int(vi) % len(marker_cycle)]
            ax.scatter(
                x[m],
                y[m],
                s=6.0,
                c=colors[m],
                alpha=0.78,
                linewidths=0,
                marker=marker,
            )
    else:
        ax.scatter(x, y, s=3.0, c=colors, alpha=0.75, linewidths=0)

    # Draw cluster centroids in the same colors, with labels, for readability.
    for c in range(int(cluster_rgb.shape[0])):
        m = sampled_clusters == c
        if not np.any(m):
            continue
        cx = float(np.median(x[m]))
        cy = float(np.median(y[m]))
        ax.scatter([cx], [cy], s=44, c=[cluster_rgb[c]], edgecolors="black", linewidths=0.5)
        ax.text(cx, cy, str(c), fontsize=7, ha="center", va="center", color="black")

    if video_labels:
        # Label one robust center per source video. This is intentionally separate
        # from cluster labels: numbers = clusters, vN/name = video provenance.
        for vi in np.unique(sampled_video_ids):
            m = sampled_video_ids == vi
            if not np.any(m):
                continue
            vx = float(np.median(x[m]))
            vy = float(np.median(y[m]))
            name = items[int(vi)].input_mp4.stem
            label = f"v{int(vi)}: {name}"
            ax.text(
                vx,
                vy,
                label,
                fontsize=7,
                ha="left",
                va="bottom",
                color="black",
                bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", pad=1.5),
            )

    if video_markers and len(items) <= int(video_legend_limit):
        handles = []
        for vi, item in enumerate(items):
            marker = marker_cycle[vi % len(marker_cycle)]
            handles.append(Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                color="black",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=6,
                label=f"v{vi}: {item.input_mp4.name}",
            ))
        ax.legend(
            handles=handles,
            title="Source video (marker shape)",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            title_fontsize=8,
            frameon=True,
        )

    ax.set_title(f"Global audio token {method.upper()} ({source_name})")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    footer = (
        f"{len(sample_idx):,}/{len(clusters):,} tokens plotted; "
        "color=NCut cluster/spectrogram color"
    )
    if video_markers:
        footer += "; marker shape/source label=video"
    if write_csv:
        footer += "; CSV gives exact point-to-video/token mapping"
    ax.text(
        0.01,
        0.01,
        footer,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Wrote global feature embedding plot: {out_png}")

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
    if args.encoder == "avclip":
        print("Loading Synchformer AVCLIP audio encoder...")
        model = load_synchformer_audio_from_avclip(
            ckpt_path=args.avclip_ckpt,
            repo_root=args.repo_root,
            device=args.device,
            half=args.half,
            max_spec_t=args.max_spec_t,
        )
    else:
        print("Loading Hugging Face AST...")
        model = load_ast(args.model_name, args.device, half=args.half)
        if args.synchformer_ckpt is not None:
            best_effort_load_synchformer_ast_weights(model, args.synchformer_ckpt.expanduser())
            model.to(args.device).eval()
            if args.half:
                model.half()

    if int(model.config.num_mel_bins) != args.num_mel_bins:
        raise RuntimeError(
            f"Model expects {model.config.num_mel_bins} mel bins, but --num-mel-bins={args.num_mel_bins}"
        )
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

    media_durations = probe_media_durations_sec(video_path)
    ffprobe_duration_sec = max(media_durations.values()) if media_durations else 0.0
    wav_duration_sec = int(waveform.shape[-1]) / float(args.sample_rate)
    fbank_duration_sec = float(fbank.shape[0]) * float(args.frame_shift_ms) / 1000.0

    # Use the largest sane duration candidate. This fixes files whose MP4
    # metadata says ~1s while ffmpeg decodes a much longer audio stream. The
    # cursor/progress bar should match the audio you actually hear.
    render_duration_sec = max(ffprobe_duration_sec, wav_duration_sec, fbank_duration_sec)
    if render_duration_sec <= 0:
        raise RuntimeError(f"Could not determine a positive render duration for {video_path}")

    duration_bits = []
    for k in sorted(media_durations):
        duration_bits.append(f"ffprobe_{k}={media_durations[k]:.3f}s")
    duration_bits.extend([
        f"decoded_wav={wav_duration_sec:.3f}s",
        f"fbank={fbank_duration_sec:.3f}s",
        f"render={render_duration_sec:.3f}s",
    ])
    print("Duration candidates for " + video_path.name + ": " + ", ".join(duration_bits))

    if any(abs(render_duration_sec - v) > 0.25 for v in [ffprobe_duration_sec, wav_duration_sec, fbank_duration_sec] if v > 0):
        print(
            f"Duration note for {video_path.name}: using the longest decoded/metadata duration "
            f"({render_duration_sec:.3f}s) for the cursor/progress bar."
        )

    print("Extracting audio patch tokens...")
    grid = extract_audio_patch_tokens_dispatch(
        model=model,
        fbank_norm=fbank_norm,
        args=args,
        desc=f"AST windows {video_path.stem}",
    )
    print(
        f"{video_path.name}: {grid.features.shape[0]:,} token nodes; "
        f"dim={grid.features.shape[1]}; render duration={render_duration_sec:.3f}s"
    )

    return VideoItem(
        input_mp4=video_path,
        output_mp4=output_path,
        wav_path=wav_path,
        fbank=fbank,
        waveform_samples=int(waveform.shape[-1]),
        sample_rate=args.sample_rate,
        duration_sec=float(render_duration_sec),
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
            "NCut on Synchformer/AST audio patch tokens. Accepts either one MP4 or a directory "
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

    p.add_argument("--encoder", choices=["avclip", "hf_ast"], default="avclip",
                   help="Audio feature source. Default: Synchformer AVCLIP audio tower. Use hf_ast for the older Hugging Face AST fallback.")
    p.add_argument("--repo-root", type=Path, default=None,
                   help="Path to the Synchformer repo root. Usually auto-detected when running from the repo.")
    p.add_argument("--avclip-ckpt", type=Path, default=None,
                   help="Synchformer segment-level AVCLIP feature-extractor .pt checkpoint. If omitted, tries to auto-find one under logs/avclip_models.")
    p.add_argument("--max-spec-t", type=int, default=66,
                   help="Spectrogram frames per Synchformer audio segment. AVCLIP checkpoints were trained with 66 by default.")
    p.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593",
                   help="HF AST model name/path, only used with --encoder hf_ast.")
    p.add_argument("--synchformer-ckpt", type=Path, default=None,
                   help="Legacy HF-AST weight remapping path, only used with --encoder hf_ast. Prefer --avclip-ckpt.")
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
    p.add_argument("--no-feature-plot", action="store_true",
                   help="Disable writing the global 2D feature scatter plot.")
    p.add_argument("--feature-plot-name", default=None,
                   help="Output PNG name/path for the global feature scatter plot. Default: joint_feature_<embedder>.png in the output directory.")
    p.add_argument("--feature-plot-source", choices=["features", "ncut"], default="features",
                   help="Points to embed in the scatter plot. 'features' plots AVCLIP/AST token features; 'ncut' plots NCut eigenvectors. Default: features.")
    p.add_argument("--feature-plot-max-points", type=int, default=50000,
                   help="Max token points to draw in the feature plot. 0 means plot all points. Default: 50000.")
    p.add_argument("--feature-plot-metric", default=None,
                   help="Distance metric for the 2D plot embedding. Default: cosine for raw features, euclidean for NCut eigenvectors.")
    p.add_argument("--feature-plot-dpi", type=int, default=180)
    p.add_argument("--no-feature-plot-video-markers", action="store_true",
                   help="Do not encode source video by marker shape in the feature plot. Cluster colors are still used.")
    p.add_argument("--no-feature-plot-video-labels", action="store_true",
                   help="Do not draw source-video centroid labels on the feature plot.")
    p.add_argument("--feature-plot-video-legend-limit", type=int, default=20,
                   help="Draw a video marker legend only when the number of videos is <= this value. Default: 20.")
    p.add_argument("--feature-plot-csv-name", default=None,
                   help="CSV name/path for per-point plot metadata. Default: <feature_plot_stem>_points.csv next to the PNG.")
    p.add_argument("--no-feature-plot-csv", action="store_true",
                   help="Disable writing the CSV that maps every plotted point to video/token coordinates.")
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
        print("Joint mode: all audio patch tokens from all videos will be concatenated before NCut.")

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

        if not args.no_feature_plot:
            if args.feature_plot_source == "features":
                plot_points = F.normalize(global_features, dim=-1).float().cpu().numpy()
                plot_source_name = "AVCLIP audio token features" if args.encoder == "avclip" else "AST audio token features"
                plot_metric = args.feature_plot_metric or "cosine"
            else:
                plot_points = eigvecs.astype(np.float32)
                plot_source_name = "NCut eigenvectors"
                plot_metric = args.feature_plot_metric or "euclidean"

            if args.feature_plot_name is None:
                feature_plot_out = (
                    output_path.parent / f"joint_feature_{args.embedder}.png"
                    if single_file_mode
                    else output_path / f"joint_feature_{args.embedder}.png"
                )
            else:
                feature_plot_out = Path(args.feature_plot_name).expanduser()
                if not feature_plot_out.is_absolute():
                    base_dir = output_path.parent if single_file_mode else output_path
                    feature_plot_out = base_dir / feature_plot_out

            if args.no_feature_plot_csv:
                feature_plot_csv_out = None
            else:
                csv_base_dir = output_path.parent if single_file_mode else output_path
                feature_plot_csv_out = _resolve_feature_plot_csv_path(
                    out_png=feature_plot_out,
                    csv_name=args.feature_plot_csv_name,
                    output_base=csv_base_dir,
                )

            write_feature_embedding_plot(
                points=plot_points,
                clusters=clusters,
                cluster_rgb=cluster_rgb,
                out_png=feature_plot_out,
                method=args.embedder,
                seed=args.seed,
                max_points=args.feature_plot_max_points,
                source_name=plot_source_name,
                metric=plot_metric,
                items=items,
                frame_shift_ms=args.frame_shift_ms,
                dpi=args.feature_plot_dpi,
                video_markers=not args.no_feature_plot_video_markers,
                video_labels=not args.no_feature_plot_video_labels,
                video_legend_limit=args.feature_plot_video_legend_limit,
                out_csv=feature_plot_csv_out,
                write_csv=not args.no_feature_plot_csv,
            )

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
