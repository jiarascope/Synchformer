"""
Synchformer dataset adapter for WebDataset-style tar shards containing MP4 files.

Place at:
    <Synchformer repo>/dataset/webdataset_tar_inmemory_cached_sync.py

Use with:
    data.dataset.target=dataset.webdataset_tar_inmemory_cached_sync.WebDatasetTarInMemoryCachedSync

Design for stage-2 Synchformer training:
    - reads MP4 bytes directly from .tar/.tar.gz/.tgz members
    - decodes video/audio in memory with PyAV; no extracted MP4s are written
    - caches FULL decoded clips per DataLoader worker BEFORE transforms
    - leaves item["targets"] empty so Synchformer's TemporalCropAndOffset transform
      performs random crop/offset during training

Important for multi-epoch training:
    - Set DataLoader persistent_workers=True, otherwise worker-local decoded caches
      are destroyed after each epoch.
    - Do not set max_clip_len_sec=5 for stage-2 training. Keep it None/null so the
      built-in transform can sample a 5s crop plus audio/video offset from the full
      10s clip.

Requirements:
    pip install av
"""

from __future__ import annotations

import copy
import glob
import io
import os
import re
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    import av
except ImportError as exc:
    raise ImportError(
        "dataset.webdataset_tar_inmemory_cached_sync requires PyAV. Install it with: pip install av"
    ) from exc

from dataset.dataset_utils import subsample_dataset


_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz")
_VIDEO_SUFFIXES = (".mp4", ".m4v", ".mov")


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def _none_if_string_null(x: Any) -> Any:
    if isinstance(x, str) and x.strip().lower() in {"", "none", "null", "~"}:
        return None
    return x


def _mapping_to_plain_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(x):
            return dict(OmegaConf.to_container(x, resolve=True))
    except Exception:
        pass
    if isinstance(x, Mapping):
        return dict(x)
    return {}


def _is_tar_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _TAR_SUFFIXES)


def _is_video_member(name: str, video_exts: Sequence[str] = _VIDEO_SUFFIXES) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in video_exts)


def _rate_to_float(rate: Any, default: float) -> float:
    if rate is None:
        return float(default)
    try:
        return float(rate)
    except Exception:
        return float(default)


def _check_close(value: float, expected: Optional[float], name: str, sample_id: str, tol: float = 1e-3) -> None:
    if expected is None:
        return
    if abs(float(value) - float(expected)) > tol:
        raise RuntimeError(f"{name} check failed for {sample_id}: got {value}, expected {expected}")


def find_tar_shards(vids_dir: str, recursive: bool = False) -> List[Path]:
    """
    Accepts:
      - directory of shards
      - one shard path
      - regular glob, e.g. /data/shards/*.tar
      - simple brace range, e.g. /data/shards/shard-{000000..000255}.tar
      - comma-separated list of any of the above
    """
    if vids_dir is None:
        return []

    pieces = [p.strip() for p in str(vids_dir).split(",") if p.strip()]
    all_shards: List[Path] = []

    for path_str in pieces:
        p = Path(path_str)

        if p.is_file() and _is_tar_path(p):
            all_shards.append(p)
            continue

        if p.is_dir():
            shards: List[Path] = []
            for pattern in ("*.tar", "*.tar.gz", "*.tgz"):
                glob_pattern = f"**/{pattern}" if recursive else pattern
                shards.extend(x for x in p.glob(glob_pattern) if x.is_file())
            all_shards.extend(shards)
            continue

        brace_match = re.search(r"\{(\d+)\.\.(\d+)\}", path_str)
        if brace_match is not None:
            start_s, end_s = brace_match.groups()
            width = max(len(start_s), len(end_s))
            start, end = int(start_s), int(end_s)
            for i in range(start, end + 1):
                expanded = path_str[: brace_match.start()] + f"{i:0{width}d}" + path_str[brace_match.end() :]
                ep = Path(expanded)
                if ep.is_file() and _is_tar_path(ep):
                    all_shards.append(ep)
            continue

        all_shards.extend(
            Path(x) for x in glob.glob(path_str) if Path(x).is_file() and _is_tar_path(Path(x))
        )

    return sorted(set(all_shards))


def decode_mp4_bytes_with_av(
    mp4_bytes: bytes,
    *,
    force_audio_rate: Optional[int] = 16000,
    force_audio_mono: bool = True,
    max_clip_len_sec: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Decode MP4 bytes from memory into Synchformer-style tensors.

    Returns:
        rgb:   uint8 tensor, shape (T, 3, H, W), range [0, 255]
        audio: float32 tensor, shape (Ta,), roughly [-1, 1]
        meta:  dict with video fps and audio framerate entries
    """
    max_clip_len_sec = _none_if_string_null(max_clip_len_sec)
    if max_clip_len_sec is not None:
        max_clip_len_sec = float(max_clip_len_sec)

    # ---- video ----
    video_container = av.open(io.BytesIO(mp4_bytes))
    try:
        video_streams = [s for s in video_container.streams if s.type == "video"]
        if not video_streams:
            raise RuntimeError("No video stream found in MP4 sample")

        video_stream = video_streams[0]
        # AUTO can use multiple codec threads where supported.
        try:
            video_stream.thread_type = "AUTO"
        except Exception:
            pass

        video_fps = _rate_to_float(video_stream.average_rate or video_stream.base_rate, 25.0)
        max_video_frames = None
        if max_clip_len_sec is not None:
            max_video_frames = int(round(max_clip_len_sec * video_fps))

        video_frames = []
        for frame_idx, frame in enumerate(video_container.decode(video_stream)):
            if max_video_frames is not None and frame_idx >= max_video_frames:
                break
            arr = frame.to_rgb().to_ndarray()  # H, W, 3, uint8
            video_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    finally:
        video_container.close()

    if not video_frames:
        raise RuntimeError("No video frames decoded from MP4 sample")
    rgb = torch.stack(video_frames, dim=0)

    # ---- audio ----
    audio_container = av.open(io.BytesIO(mp4_bytes))
    try:
        audio_streams = [s for s in audio_container.streams if s.type == "audio"]
        if not audio_streams:
            raise RuntimeError("No audio stream found in MP4 sample")

        audio_stream = audio_streams[0]
        src_audio_rate = int(audio_stream.rate or force_audio_rate or 16000)
        out_audio_rate = int(force_audio_rate or src_audio_rate)

        resampler = None
        if force_audio_rate is not None or force_audio_mono:
            resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout="mono" if force_audio_mono else None,
                rate=out_audio_rate,
            )

        max_audio_samples = None
        if max_clip_len_sec is not None:
            max_audio_samples = int(round(max_clip_len_sec * out_audio_rate))

        audio_chunks = []
        total_audio_samples = 0

        def append_audio_frame(out_frame: Any) -> None:
            nonlocal total_audio_samples
            arr = out_frame.to_ndarray()  # usually channels, samples
            tensor = torch.from_numpy(arr).float()
            if tensor.numel() > 0 and tensor.abs().max() > 2.0:
                tensor = tensor / 32768.0
            if tensor.ndim == 2:
                tensor = tensor.mean(dim=0)
            elif tensor.ndim != 1:
                tensor = tensor.reshape(-1)
            tensor = tensor.contiguous()
            audio_chunks.append(tensor)
            total_audio_samples += int(tensor.numel())

        for frame in audio_container.decode(audio_stream):
            frames = resampler.resample(frame) if resampler is not None else [frame]
            for out_frame in frames:
                append_audio_frame(out_frame)
                if max_audio_samples is not None and total_audio_samples >= max_audio_samples:
                    break
            if max_audio_samples is not None and total_audio_samples >= max_audio_samples:
                break

        if resampler is not None:
            for out_frame in resampler.resample(None):
                append_audio_frame(out_frame)
                if max_audio_samples is not None and total_audio_samples >= max_audio_samples:
                    break
    finally:
        audio_container.close()

    if not audio_chunks:
        raise RuntimeError("No audio decoded from MP4 sample")

    audio = torch.cat(audio_chunks, dim=0).float().contiguous()
    if max_audio_samples is not None:
        audio = audio[:max_audio_samples]

    meta = {
        "video": {"fps": [video_fps], "duration": [float(rgb.shape[0]) / float(video_fps)]},
        "audio": {"framerate": [out_audio_rate], "duration": [float(audio.shape[0]) / float(out_audio_rate)]},
    }
    return rgb, audio, meta


@dataclass
class DecodedSample:
    sample_id: str
    rgb: torch.Tensor
    audio: torch.Tensor
    meta: dict


class WebDatasetTarInMemoryCachedSync(torch.utils.data.Dataset):
    """
    Map-style Dataset for naturally synchronized MP4 clips inside tar shards.

    Compatible with Synchformer's old get_datasets() signature. For split-specific
    paths and cache-size CLI overrides, apply the companion train_utils patch so
    data.dataset.params.* are forwarded to this constructor.
    """

    def __init__(
        self,
        split: str,
        vids_dir: str,
        transforms=None,
        to_filter_bad_examples: bool = False,   # accepted for compatibility
        splits_path: str = "./data",            # accepted for compatibility
        meta_path: Optional[str] = None,         # accepted for compatibility
        seed: int = 1337,
        load_fixed_offsets_on: Optional[Sequence[str]] = None,
        vis_load_backend: str = "pyav_inmemory",
        size_ratio: Optional[float] = None,
        attr_annot_path: Optional[str] = None,   # accepted for compatibility
        max_attr_per_vid: Optional[int] = None,  # accepted for compatibility
        recursive: bool = False,
        train_vids_dir: Optional[str] = None,
        valid_vids_dir: Optional[str] = None,
        test_vids_dir: Optional[str] = None,
        split_paths: Optional[Mapping[str, str]] = None,
        video_exts: Sequence[str] = _VIDEO_SUFFIXES,
        audio_rate: int = 16000,
        audio_mono: bool = True,
        max_clip_len_sec: Optional[float] = None,
        strict_video_fps: Optional[float] = None,
        strict_audio_fps: Optional[float] = None,
        cache_decoded: bool = True,
        decoded_cache_size: int = 64,
        cache_tar_handles: bool = True,
        tar_handle_cache_size: int = 8,
        clone_cached_tensors: bool = False,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__()
        self.split = split
        self.transforms = transforms
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio
        self.recursive = _as_bool(recursive)
        self.video_exts = tuple(video_exts)
        self.audio_rate = int(audio_rate)
        self.audio_mono = _as_bool(audio_mono)
        self.max_clip_len_sec = _none_if_string_null(max_clip_len_sec)
        self.strict_video_fps = None if _none_if_string_null(strict_video_fps) is None else float(strict_video_fps)
        self.strict_audio_fps = None if _none_if_string_null(strict_audio_fps) is None else float(strict_audio_fps)
        self.cache_decoded = _as_bool(cache_decoded)
        self.decoded_cache_size = int(decoded_cache_size)
        self.cache_tar_handles = _as_bool(cache_tar_handles)
        self.tar_handle_cache_size = int(tar_handle_cache_size)
        self.clone_cached_tensors = _as_bool(clone_cached_tensors)

        split_paths_dict = _mapping_to_plain_dict(split_paths)
        split_specific = {
            "train": train_vids_dir,
            "valid": valid_vids_dir,
            "val": valid_vids_dir,
            "test": test_vids_dir,
        }.get(split)
        self.vids_dir = str(_none_if_string_null(split_specific) or split_paths_dict.get(split) or vids_dir)

        self._decoded_cache: OrderedDict[Tuple[str, str], DecodedSample] = OrderedDict()
        self._tar_handles: OrderedDict[str, tarfile.TarFile] = OrderedDict()

        shards = find_tar_shards(self.vids_dir, recursive=self.recursive)
        if len(shards) == 0:
            raise RuntimeError(f"No tar shards found for split={split!r} from vids_dir={self.vids_dir!r}")
        self.shards = shards

        samples = self._build_index(shards)
        if len(samples) == 0:
            raise RuntimeError(
                f"No video members {self.video_exts} found inside {len(shards)} shards from {self.vids_dir!r}"
            )

        self.dataset = subsample_dataset(samples, size_ratio, shuffle=(split == "train"))

    def __getstate__(self):
        # DataLoader workers pickle the dataset. Never pickle open tar handles/cached tensors.
        state = self.__dict__.copy()
        state["_decoded_cache"] = OrderedDict()
        state["_tar_handles"] = OrderedDict()
        return state

    def close(self) -> None:
        for tar in self._tar_handles.values():
            try:
                tar.close()
            except Exception:
                pass
        self._tar_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_index(self, shards: Sequence[Path]) -> List[Tuple[str, str]]:
        samples: List[Tuple[str, str]] = []
        for shard in shards:
            with tarfile.open(shard, "r:*") as tar:
                for member in tar:
                    if member.isfile() and _is_video_member(member.name, self.video_exts):
                        samples.append((str(shard), member.name))
        return samples

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_tar(self, shard_path: str) -> tarfile.TarFile:
        if not self.cache_tar_handles:
            return tarfile.open(shard_path, "r:*")

        cached = self._tar_handles.get(shard_path)
        if cached is not None:
            self._tar_handles.move_to_end(shard_path)
            return cached

        tar = tarfile.open(shard_path, "r:*")
        self._tar_handles[shard_path] = tar

        if self.tar_handle_cache_size > 0:
            while len(self._tar_handles) > self.tar_handle_cache_size:
                _, old_tar = self._tar_handles.popitem(last=False)
                old_tar.close()
        return tar

    def _read_mp4_bytes(self, shard_path: str, member_name: str, sample_id: str) -> bytes:
        if self.cache_tar_handles:
            tar = self._get_tar(shard_path)
            member = tar.getmember(member_name)
            src = tar.extractfile(member)
            if src is None:
                raise RuntimeError(f"Could not read {sample_id}")
            return src.read()

        with self._get_tar(shard_path) as tar:
            member = tar.getmember(member_name)
            src = tar.extractfile(member)
            if src is None:
                raise RuntimeError(f"Could not read {sample_id}")
            return src.read()

    def _load_and_decode_uncached(self, index: int) -> DecodedSample:
        shard_path, member_name = self.dataset[index]
        sample_id = f"{shard_path}::{member_name}"
        mp4_bytes = self._read_mp4_bytes(shard_path, member_name, sample_id)

        rgb, audio, meta = decode_mp4_bytes_with_av(
            mp4_bytes,
            force_audio_rate=self.audio_rate,
            force_audio_mono=self.audio_mono,
            max_clip_len_sec=self.max_clip_len_sec,
        )

        _check_close(float(meta["video"]["fps"][0]), self.strict_video_fps, "video fps", sample_id)
        _check_close(float(meta["audio"]["framerate"][0]), self.strict_audio_fps, "audio fps", sample_id)
        return DecodedSample(sample_id=sample_id, rgb=rgb, audio=audio, meta=meta)

    def _get_decoded(self, index: int) -> DecodedSample:
        if not self.cache_decoded:
            return self._load_and_decode_uncached(index)

        shard_path, member_name = self.dataset[index]
        key = (str(shard_path), str(member_name))
        cached = self._decoded_cache.get(key)
        if cached is not None:
            self._decoded_cache.move_to_end(key)
            return cached

        value = self._load_and_decode_uncached(index)
        self._decoded_cache[key] = value

        # decoded_cache_size <= 0 means unlimited. Be careful: decoded RGB/audio is large.
        if self.decoded_cache_size > 0:
            while len(self._decoded_cache) > self.decoded_cache_size:
                self._decoded_cache.popitem(last=False)
        return value

    def __getitem__(self, index: int):
        decoded = self._get_decoded(index)

        # Usually safe to leave False because Synchformer transforms assign new tensors/slices.
        # Set True only if you add in-place transforms that mutate raw decoded RGB/audio.
        rgb = decoded.rgb.clone() if self.clone_cached_tensors else decoded.rgb
        audio = decoded.audio.clone() if self.clone_cached_tensors else decoded.audio

        item = {
            "video": rgb,
            "audio": audio,
            "meta": copy.deepcopy(decoded.meta),
            "path": decoded.sample_id,
            "targets": {},  # critical: TemporalCropAndOffset samples random train offsets/crops
            "split": self.split,
        }

        if self.transforms is not None:
            item = self.transforms(item)
        return item
