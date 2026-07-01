from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .audio_tokens import VideoItem


def parse_cluster_id_list(text: str | Sequence[str] | None) -> list[int]:
    """Parse cluster ids from CLI forms like `3,7,12`, `3 7 12`, or `3, 7, 12`."""
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        raw = " ".join(str(x) for x in text)
    else:
        raw = str(text)
    cleaned = raw.replace(",", " ").strip()
    if not cleaned:
        return []
    ids: list[int] = []
    for part in cleaned.split():
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Cluster id must be an integer, got {part!r} in {raw!r}") from exc
        if value < 0:
            raise ValueError(f"Cluster ids must be non-negative, got {value}")
        ids.append(value)
    # Preserve user order while removing duplicates.
    return list(dict.fromkeys(ids))


def cluster_ids_slug(cluster_ids: Sequence[int]) -> str:
    if not cluster_ids:
        return "none"
    return "_".join(str(int(c)) for c in cluster_ids)


def rgb_to_hex(rgb: Sequence[float]) -> str:
    vals = [int(round(float(v) * 255.0)) for v in rgb]
    vals = [max(0, min(255, v)) for v in vals]
    return "#" + "".join(f"{v:02x}" for v in vals)


def resolve_metadata_root(output_path: Path, single_file_mode: bool, token_metadata_dir: Path | None) -> Path:
    """Return the shared metadata root used by many runs.

    By default, metadata is stored beside the output run directory instead of
    directly inside it, so sibling output runs can find each other's parent
    metadata without requiring --metadata-cache.
    """
    if token_metadata_dir is not None:
        return token_metadata_dir.expanduser().resolve()
    base = output_path.parent if single_file_mode else output_path.parent
    return (base / "ncut_metadata").resolve()


def resolve_metadata_dir(output_path: Path, single_file_mode: bool, token_metadata_dir: Path | None) -> Path:
    """Backward-compatible alias for old callers."""
    return resolve_metadata_root(output_path, single_file_mode, token_metadata_dir)


def _safe_slug(text: str, max_len: int = 96) -> str:
    text = text.replace(str(Path.home()), "~")
    text = re.sub(r"[^A-Za-z0-9._=-]+", "__", text).strip("._-")
    return (text or "input")[:max_len]


def _input_slug(input_paths: Sequence[Path]) -> str:
    parts: list[str] = []
    cwd = Path.cwd().resolve()
    for path in input_paths:
        p = Path(path).expanduser().resolve()
        try:
            rel = p.relative_to(cwd)
        except ValueError:
            rel = p
        parts.append(str(rel))
    return _safe_slug("--".join(parts), max_len=120)


def _run_hash(videos: Sequence[Path], args: Any, parent_cluster_ids: Sequence[int] | None = None) -> str:
    h = hashlib.sha1()
    for video in videos:
        p = Path(video).expanduser().resolve()
        h.update(str(p).encode("utf-8", "surrogatepass"))
        try:
            st = p.stat()
            h.update(f"|{int(st.st_size)}|{int(st.st_mtime)}".encode())
        except OSError:
            pass
        h.update(b"\0")
    keys = [
        "ncut_mode", "n_eig", "n_clusters", "ncut_clusterer", "seed",
        "encoder", "sample_rate", "num_mel_bins", "frame_shift_ms",
        "chunk_hop_frames", "max_spec_t", "embedder",
    ]
    for key in keys:
        h.update(f"{key}={getattr(args, key, None)};".encode())
    if parent_cluster_ids:
        h.update(("parent=" + cluster_ids_slug(parent_cluster_ids)).encode())
    return h.hexdigest()[:10]


def build_metadata_run_name(
    input_paths: Sequence[Path],
    videos: Sequence[Path],
    args: Any,
    parent_cluster_ids: Sequence[int] | None = None,
) -> str:
    parent_part = ""
    if parent_cluster_ids:
        parent_part = f"__parent_{cluster_ids_slug(parent_cluster_ids)}"
    hop = getattr(args, "chunk_hop_frames", None)
    hop_part = "none" if hop is None else str(hop)
    name = (
        f"{_input_slug(input_paths)}"
        f"__eig{int(getattr(args, 'n_eig'))}"
        f"__clusters{int(getattr(args, 'n_clusters'))}"
        f"__hop{hop_part}"
        f"__{getattr(args, 'encoder', 'encoder')}"
        f"__{getattr(args, 'ncut_clusterer', 'clusterer')}"
        f"__seed{int(getattr(args, 'seed', 0))}"
        f"__sr{int(getattr(args, 'sample_rate', 0))}"
        f"__mel{int(getattr(args, 'num_mel_bins', 0))}"
        f"__{getattr(args, 'ncut_mode', 'mode')}"
        f"{parent_part}"
        f"__{_run_hash(videos, args, parent_cluster_ids)}"
    )
    return _safe_slug(name, max_len=220)


def resolve_run_metadata_dir(
    metadata_root: Path,
    input_paths: Sequence[Path],
    videos: Sequence[Path],
    args: Any,
    parent_cluster_ids: Sequence[int] | None = None,
    flat: bool = False,
) -> Path:
    metadata_root = Path(metadata_root).expanduser().resolve()
    if flat:
        return metadata_root
    return metadata_root / build_metadata_run_name(input_paths, videos, args, parent_cluster_ids)


def token_cache_path(metadata_dir: Path, prefix: str) -> Path:
    return metadata_dir / f"{prefix}_token_cache.npz"


def build_global_token_metadata(items: Sequence[VideoItem]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-token video ids, local token ids, and spectrogram coords."""
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
                f"Internal token offset mismatch for {item.input_mp4}: offsets give {n}, "
                f"grid has {item.token_count}"
            )
        video_ids[s:e] = vi
        local_token_ids[s:e] = np.arange(n, dtype=np.int64)
        coords[s:e] = item.grid.coords.astype(np.int32, copy=False)
    return video_ids, local_token_ids, coords


def _item_arrays(items: Sequence[VideoItem]) -> dict[str, np.ndarray]:
    return {
        "video_paths": np.asarray([str(item.input_mp4) for item in items], dtype=np.str_),
        "video_names": np.asarray([item.input_mp4.name for item in items], dtype=np.str_),
        "output_paths": np.asarray([str(item.output_mp4) for item in items], dtype=np.str_),
        "video_token_start": np.asarray([int(item.token_start) for item in items], dtype=np.int64),
        "video_token_end": np.asarray([int(item.token_end) for item in items], dtype=np.int64),
        "video_token_count": np.asarray([int(item.token_count) for item in items], dtype=np.int64),
        "duration_sec": np.asarray([float(item.duration_sec) for item in items], dtype=np.float64),
        "fbank_num_frames": np.asarray([int(item.fbank.shape[0]) for item in items], dtype=np.int64),
        "fbank_num_mel_bins": np.asarray([int(item.fbank.shape[1]) for item in items], dtype=np.int64),
        "sample_rate": np.asarray([int(item.sample_rate) for item in items], dtype=np.int32),
    }


def _clusters_for_npz(clusters: np.ndarray, ignore_label: int = -1) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for c in sorted(int(x) for x in np.unique(clusters) if int(x) != int(ignore_label)):
        out[f"cluster_{c}"] = np.flatnonzero(clusters == c).astype(np.int64)
    return out


def write_cluster_token_indices_npz(out_npz: Path, clusters: np.ndarray, ignore_label: int = -1) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = _clusters_for_npz(np.asarray(clusters, dtype=np.int32), ignore_label=ignore_label)
    np.savez_compressed(out_npz, **arrays)
    print(f"Wrote cluster token-index mapping: {out_npz}")


def write_token_metadata_csv(
    out_csv: Path,
    items: Sequence[VideoItem],
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    video_ids: np.ndarray,
    local_token_ids: np.ndarray,
    coords: np.ndarray,
    frame_shift_ms: float,
    parent_clusters: np.ndarray | None = None,
    selected_global_indices: np.ndarray | None = None,
    include_unassigned: bool = False,
    ignore_label: int = -1,
) -> None:
    """Write one CSV row per assigned token with exact token patch location and color."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clusters = np.asarray(clusters, dtype=np.int32)
    selected_lookup: set[int] | None = None
    if selected_global_indices is not None:
        selected_lookup = set(int(i) for i in np.asarray(selected_global_indices, dtype=np.int64))

    frame_to_sec = float(frame_shift_ms) / 1000.0
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_token_index",
            "video_index",
            "video_name",
            "video_path",
            "local_token_index",
            "cluster_id",
            "parent_cluster_id",
            "selected_for_this_stage",
            "color_r",
            "color_g",
            "color_b",
            "color_hex",
            "freq_bin_start",
            "freq_bin_end",
            "time_frame_start",
            "time_frame_end",
            "time_sec_start",
            "time_sec_end",
        ])
        for global_idx, c_raw in enumerate(clusters):
            c = int(c_raw)
            if c == int(ignore_label) and not include_unassigned:
                continue
            vi = int(video_ids[global_idx])
            item = items[vi]
            f0, f1, t0, t1 = [int(v) for v in coords[global_idx]]
            if c == int(ignore_label):
                rgb = np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
                hex_color = ""
            else:
                rgb = np.asarray(cluster_rgb[c], dtype=np.float32)
                hex_color = rgb_to_hex(rgb)
            parent = "" if parent_clusters is None else int(parent_clusters[global_idx])
            selected = "" if selected_lookup is None else int(global_idx in selected_lookup)
            writer.writerow([
                int(global_idx),
                vi,
                item.input_mp4.name,
                str(item.input_mp4),
                int(local_token_ids[global_idx]),
                c,
                parent,
                selected,
                float(rgb[0]),
                float(rgb[1]),
                float(rgb[2]),
                hex_color,
                f0,
                f1,
                t0,
                t1,
                float(t0 * frame_to_sec),
                float(t1 * frame_to_sec),
            ])
    print(f"Wrote token metadata CSV: {out_csv}")


def write_cluster_summary_csv(
    out_csv: Path,
    items: Sequence[VideoItem],
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    frame_shift_ms: float,
    parent_clusters: np.ndarray | None = None,
    ignore_label: int = -1,
) -> None:
    """Write cluster color and coarse token support, overall and per video."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clusters = np.asarray(clusters, dtype=np.int32)
    frame_to_sec = float(frame_shift_ms) / 1000.0

    def row_for(cluster_id: int, video_index: int, video_name: str, video_path: str, token_indices: np.ndarray) -> list[Any]:
        # Build from global token indices so the overall row and per-video rows use identical logic.
        coords_list = []
        parent_ids: list[int] = []
        for global_idx in token_indices:
            vi = int(_video_ids_cache[global_idx])
            local_idx = int(_local_ids_cache[global_idx])
            coords_list.append(items[vi].grid.coords[local_idx])
            if parent_clusters is not None:
                parent_ids.append(int(parent_clusters[global_idx]))
        arr = np.asarray(coords_list, dtype=np.int32)
        f0 = int(arr[:, 0].min())
        f1 = int(arr[:, 1].max())
        t0 = int(arr[:, 2].min())
        t1 = int(arr[:, 3].max())
        rgb = np.asarray(cluster_rgb[cluster_id], dtype=np.float32)
        parent_set = "" if parent_clusters is None else " ".join(str(x) for x in sorted(set(parent_ids)))
        return [
            int(cluster_id),
            float(rgb[0]),
            float(rgb[1]),
            float(rgb[2]),
            rgb_to_hex(rgb),
            int(video_index),
            video_name,
            video_path,
            int(len(token_indices)),
            parent_set,
            f0,
            f1,
            t0,
            t1,
            float(t0 * frame_to_sec),
            float(t1 * frame_to_sec),
        ]

    _video_ids_cache, _local_ids_cache, _coords_cache = build_global_token_metadata(items)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id",
            "color_r",
            "color_g",
            "color_b",
            "color_hex",
            "video_index",
            "video_name",
            "video_path",
            "token_count",
            "parent_cluster_ids_present",
            "freq_bin_start",
            "freq_bin_end",
            "time_frame_start",
            "time_frame_end",
            "time_sec_start",
            "time_sec_end",
        ])
        for c in sorted(int(x) for x in np.unique(clusters) if int(x) != int(ignore_label)):
            all_idx = np.flatnonzero(clusters == c).astype(np.int64)
            if len(all_idx) == 0:
                continue
            writer.writerow(row_for(c, -1, "ALL", "", all_idx))
            for vi, item in enumerate(items):
                s, e = int(item.token_start), int(item.token_end)
                local = np.flatnonzero(clusters[s:e] == c).astype(np.int64)
                if len(local) == 0:
                    continue
                global_idx = local + s
                writer.writerow(row_for(c, vi, item.input_mp4.name, str(item.input_mp4), global_idx))
    print(f"Wrote cluster summary CSV: {out_csv}")


def write_token_cache_npz(
    out_npz: Path,
    items: Sequence[VideoItem],
    global_features: torch.Tensor,
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    video_ids: np.ndarray,
    local_token_ids: np.ndarray,
    coords: np.ndarray,
    frame_shift_ms: float,
    stage_name: str,
    save_features: bool = True,
    eigvecs: np.ndarray | None = None,
    parent_clusters: np.ndarray | None = None,
    selected_parent_cluster_ids: Sequence[int] | None = None,
    selected_global_indices: np.ndarray | None = None,
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "clusters": np.asarray(clusters, dtype=np.int32),
        "cluster_rgb": np.asarray(cluster_rgb, dtype=np.float32),
        "coords": np.asarray(coords, dtype=np.int32),
        "video_ids": np.asarray(video_ids, dtype=np.int32),
        "local_token_ids": np.asarray(local_token_ids, dtype=np.int64),
        "frame_shift_ms": np.asarray([float(frame_shift_ms)], dtype=np.float64),
        "stage_name": np.asarray([stage_name], dtype=np.str_),
        **_item_arrays(items),
    }
    if save_features:
        arrays["features"] = global_features.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if eigvecs is not None:
        arrays["eigvecs"] = np.asarray(eigvecs, dtype=np.float32)
    if parent_clusters is not None:
        arrays["parent_clusters"] = np.asarray(parent_clusters, dtype=np.int32)
    if selected_parent_cluster_ids is not None:
        arrays["selected_parent_cluster_ids"] = np.asarray(selected_parent_cluster_ids, dtype=np.int32)
    if selected_global_indices is not None:
        arrays["selected_global_indices"] = np.asarray(selected_global_indices, dtype=np.int64)
    np.savez_compressed(out_npz, **arrays)
    print(f"Wrote token/cache NPZ: {out_npz}")


def write_manifest_json(
    out_json: Path,
    paths: Mapping[str, Path],
    items: Sequence[VideoItem],
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    args_dict: Mapping[str, Any],
    stage_name: str,
    parent_cache: Path | None = None,
    selected_parent_cluster_ids: Sequence[int] | None = None,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    unique, counts = np.unique(clusters[clusters >= 0], return_counts=True)
    cluster_counts = {str(int(c)): int(n) for c, n in zip(unique, counts)}
    manifest = {
        "stage_name": stage_name,
        "num_videos": len(items),
        "num_tokens_total": int(len(clusters)),
        "num_tokens_assigned": int(np.sum(clusters >= 0)),
        "cluster_counts": cluster_counts,
        "cluster_colors": {
            str(i): {
                "rgb": [float(v) for v in cluster_rgb[i]],
                "hex": rgb_to_hex(cluster_rgb[i]),
            }
            for i in range(int(cluster_rgb.shape[0]))
        },
        "videos": [
            {
                "video_index": i,
                "video_name": item.input_mp4.name,
                "video_path": str(item.input_mp4),
                "token_start": int(item.token_start),
                "token_end": int(item.token_end),
                "token_count": int(item.token_count),
                "duration_sec": float(item.duration_sec),
                "fbank_shape": [int(item.fbank.shape[0]), int(item.fbank.shape[1])],
            }
            for i, item in enumerate(items)
        ],
        "paths": {k: str(v) for k, v in paths.items()},
        "parent_cache": None if parent_cache is None else str(parent_cache),
        "selected_parent_cluster_ids": None if selected_parent_cluster_ids is None else [int(x) for x in selected_parent_cluster_ids],
        "args": dict(args_dict),
    }
    out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote metadata manifest: {out_json}")


def _jsonable_args(args: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def write_ncut_metadata_bundle(
    metadata_dir: Path,
    prefix: str,
    stage_name: str,
    items: Sequence[VideoItem],
    global_features: torch.Tensor,
    clusters: np.ndarray,
    cluster_rgb: np.ndarray,
    frame_shift_ms: float,
    args: Any,
    save_features: bool = True,
    save_eigvecs: bool = False,
    eigvecs: np.ndarray | None = None,
    parent_clusters: np.ndarray | None = None,
    parent_cache: Path | None = None,
    selected_parent_cluster_ids: Sequence[int] | None = None,
    selected_global_indices: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write metadata needed to inspect and reuse token/cluster assignments."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    clusters = np.asarray(clusters, dtype=np.int32)
    cluster_rgb = np.asarray(cluster_rgb, dtype=np.float32)
    video_ids, local_token_ids, coords = build_global_token_metadata(items)

    paths = {
        "token_cache_npz": token_cache_path(metadata_dir, prefix),
        "token_metadata_csv": metadata_dir / f"{prefix}_token_metadata.csv",
        "cluster_summary_csv": metadata_dir / f"{prefix}_cluster_summary.csv",
        "cluster_token_indices_npz": metadata_dir / f"{prefix}_cluster_token_indices.npz",
        "manifest_json": metadata_dir / f"{prefix}_metadata_manifest.json",
    }
    write_token_cache_npz(
        out_npz=paths["token_cache_npz"],
        items=items,
        global_features=global_features,
        clusters=clusters,
        cluster_rgb=cluster_rgb,
        video_ids=video_ids,
        local_token_ids=local_token_ids,
        coords=coords,
        frame_shift_ms=frame_shift_ms,
        stage_name=stage_name,
        save_features=save_features,
        eigvecs=eigvecs if save_eigvecs else None,
        parent_clusters=parent_clusters,
        selected_parent_cluster_ids=selected_parent_cluster_ids,
        selected_global_indices=selected_global_indices,
    )
    write_token_metadata_csv(
        out_csv=paths["token_metadata_csv"],
        items=items,
        clusters=clusters,
        cluster_rgb=cluster_rgb,
        video_ids=video_ids,
        local_token_ids=local_token_ids,
        coords=coords,
        frame_shift_ms=frame_shift_ms,
        parent_clusters=parent_clusters,
        selected_global_indices=selected_global_indices,
        include_unassigned=False,
    )
    write_cluster_summary_csv(
        out_csv=paths["cluster_summary_csv"],
        items=items,
        clusters=clusters,
        cluster_rgb=cluster_rgb,
        frame_shift_ms=frame_shift_ms,
        parent_clusters=parent_clusters,
    )
    write_cluster_token_indices_npz(paths["cluster_token_indices_npz"], clusters)
    write_manifest_json(
        out_json=paths["manifest_json"],
        paths=paths,
        items=items,
        clusters=clusters,
        cluster_rgb=cluster_rgb,
        args_dict=_jsonable_args(args),
        stage_name=stage_name,
        parent_cache=parent_cache,
        selected_parent_cluster_ids=selected_parent_cluster_ids,
    )
    return paths


def _manifest_video_paths(manifest_path: Path) -> list[str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    videos = manifest.get("videos", [])
    return [str(v.get("video_path", "")) for v in videos]


def _manifest_args(manifest_path: Path) -> Mapping[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    args = manifest.get("args", {})
    return args if isinstance(args, Mapping) else {}


def find_matching_parent_cache(
    metadata_root: Path,
    prefix: str,
    items: Sequence[VideoItem],
    current_metadata_dir: Path | None = None,
    parent_n_eig: int | None = None,
    parent_n_clusters: int | None = None,
    parent_seed: int | None = None,
) -> Path:
    """Find a previous stage-1 cache under metadata_root matching the current videos."""
    metadata_root = Path(metadata_root).expanduser().resolve()
    if not metadata_root.exists():
        raise FileNotFoundError(
            f"Metadata root does not exist: {metadata_root}. Pass --metadata-cache explicitly "
            "or run the parent/global NCut once without --no-token-feature-cache."
        )
    current_paths = [str(item.input_mp4) for item in items]
    candidates: list[Path] = []
    pattern = f"**/{prefix}_metadata_manifest.json"
    for manifest_path in metadata_root.glob(pattern):
        if current_metadata_dir is not None:
            try:
                if manifest_path.parent.resolve() == Path(current_metadata_dir).resolve():
                    continue
            except OSError:
                pass
        if _manifest_video_paths(manifest_path) != current_paths:
            continue
        args = _manifest_args(manifest_path)
        if parent_n_eig is not None and int(args.get("n_eig", -1)) != int(parent_n_eig):
            continue
        if parent_n_clusters is not None and int(args.get("n_clusters", -1)) != int(parent_n_clusters):
            continue
        if parent_seed is not None and int(args.get("seed", -1)) != int(parent_seed):
            continue
        cache_path = manifest_path.parent / f"{prefix}_token_cache.npz"
        if cache_path.exists():
            candidates.append(cache_path)
    if not candidates:
        raise FileNotFoundError(
            "No matching parent metadata cache found under "
            f"{metadata_root}. Pass --metadata-cache /path/to/{prefix}_token_cache.npz, "
            "or set --token-metadata-dir to the shared metadata root used by the parent run."
        )
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        msg = "Multiple matching parent metadata caches found:\n" + "\n".join(f"  {p}" for p in candidates[:20])
        msg += "\nDisambiguate with --metadata-cache, --parent-n-eig, --parent-n-clusters, or --parent-seed."
        raise RuntimeError(msg)
    return candidates[0].resolve()


def load_token_cache(cache_path: Path) -> dict[str, np.ndarray]:
    cache_path = cache_path.expanduser().resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Token metadata cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    required = {"clusters", "coords", "video_ids", "local_token_ids", "video_names"}
    missing = sorted(required - set(out))
    if missing:
        raise ValueError(f"Token cache {cache_path} is missing required arrays: {missing}")
    return out


def select_cluster_token_indices(parent_clusters: np.ndarray, selected_cluster_ids: Sequence[int]) -> np.ndarray:
    parent_clusters = np.asarray(parent_clusters, dtype=np.int32)
    selected_cluster_ids = [int(c) for c in selected_cluster_ids]
    if not selected_cluster_ids:
        raise ValueError("No parent clusters were selected.")
    available = set(int(c) for c in np.unique(parent_clusters) if int(c) >= 0)
    missing = [c for c in selected_cluster_ids if c not in available]
    if missing:
        raise ValueError(
            f"Selected parent cluster id(s) not present in metadata: {missing}. "
            f"Available clusters: {sorted(available)}"
        )
    selected = np.flatnonzero(np.isin(parent_clusters, np.asarray(selected_cluster_ids, dtype=np.int32))).astype(np.int64)
    if len(selected) == 0:
        raise ValueError(f"Selected clusters {selected_cluster_ids} contain no tokens.")
    return selected


def validate_token_cache_against_items(
    cache: Mapping[str, np.ndarray],
    items: Sequence[VideoItem],
    allow_mismatch: bool = False,
) -> None:
    """Validate that cached stage-1 token rows still align with this run's extracted token rows."""
    video_ids, local_token_ids, coords = build_global_token_metadata(items)
    errors: list[str] = []
    warnings: list[str] = []

    cached_clusters = np.asarray(cache["clusters"])
    if cached_clusters.shape[0] != coords.shape[0]:
        errors.append(f"token count differs: cache has {cached_clusters.shape[0]}, current run has {coords.shape[0]}")
    cached_coords = np.asarray(cache["coords"])
    if cached_coords.shape != coords.shape:
        errors.append(f"coords shape differs: cache has {cached_coords.shape}, current run has {coords.shape}")
    elif not np.array_equal(cached_coords, coords):
        errors.append("token spectrogram coordinates differ between cache and current run")

    cached_video_ids = np.asarray(cache["video_ids"])
    if cached_video_ids.shape != video_ids.shape or not np.array_equal(cached_video_ids, video_ids):
        errors.append("video_id mapping differs between cache and current run")
    cached_local_ids = np.asarray(cache["local_token_ids"])
    if cached_local_ids.shape != local_token_ids.shape or not np.array_equal(cached_local_ids, local_token_ids):
        errors.append("local_token_id mapping differs between cache and current run")

    cached_names = [str(x) for x in np.asarray(cache.get("video_names", []))]
    current_names = [item.input_mp4.name for item in items]
    if cached_names and cached_names != current_names:
        warnings.append(f"video names differ: cache={cached_names}, current={current_names}")

    cached_paths = [str(x) for x in np.asarray(cache.get("video_paths", []))]
    current_paths = [str(item.input_mp4) for item in items]
    if cached_paths and cached_paths != current_paths:
        warnings.append("video paths differ between cache and current run")

    for warning in warnings:
        print(f"WARNING: metadata validation: {warning}")
    if errors:
        msg = "Metadata cache does not align with the current token extraction:\n  - " + "\n  - ".join(errors)
        if not allow_mismatch:
            msg += "\nUse --allow-metadata-mismatch only if you are certain token row order is still identical."
            raise ValueError(msg)
        print("WARNING: " + msg)
    else:
        print("Metadata cache validation passed: cached token rows align with current extraction.")
