from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from .audio_media import extract_audio_to_wav, probe_media_durations_sec


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

