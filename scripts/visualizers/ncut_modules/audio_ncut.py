from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .audio_tokens import VideoItem
from .ncut_ops import run_ncut as run_ncut_eigenvectors


def _print_cluster_diagnostics(name: str, clusters: np.ndarray, requested_clusters: int) -> None:
    unique, counts = np.unique(clusters, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = ", ".join(f"{int(unique[j])}:{int(counts[j])}" for j in order[:12])
    print(
        f"{name}: requested_clusters={requested_clusters}, "
        f"actual_unique_clusters={len(unique)}, tokens={len(clusters):,}"
    )
    print(f"{name}: largest cluster sizes: {top}")
    if len(unique) != int(requested_clusters):
        print(
            f"WARNING: {name}: requested {requested_clusters} clusters but got "
            f"{len(unique)} non-empty labels. This can happen if the clustering backend "
            "returns empty clusters or if the data/eigenvectors collapse."
        )


def run_ncut(
    features: torch.Tensor,
    n_eig: int,
    n_clusters: int,
    device: str,
    seed: int = 0,
    clusterer: str = "kmeans",
    name: str = "NCut",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Nyström NCut and convert eigenvectors to discrete labels.

    Important: ncut_pytorch's documented kway_ncut API uses the number of
    eigenvector columns as the requested number of segments, rather than a
    stable n_clusters keyword. To make --n-clusters unambiguous, the default
    here is sklearn KMeans/MiniBatchKMeans on the NCut eigenvectors. Pass
    --ncut-clusterer kway to use ncut_pytorch's own discretizer.
    """
    n_eig = int(n_eig)
    n_clusters = int(n_clusters)
    if n_eig < 1 or n_clusters < 1:
        raise ValueError(f"n_eig and n_clusters must be positive; got {n_eig}, {n_clusters}")
    if n_eig < n_clusters:
        print(
            f"WARNING: {name}: --n-eig ({n_eig}) is smaller than --n-clusters ({n_clusters}). "
            f"Only {n_eig} NCut dimensions are available for clustering."
        )

    x = F.normalize(features.to(device), dim=-1)
    print(f"{name}: running Ncut(n_eig={n_eig}) on {x.shape[0]:,} tokens...")
    eigvecs = run_ncut_eigenvectors(x, num_eig=n_eig, device=torch.device(device))
    if int(eigvecs.shape[1]) != n_eig:
        raise RuntimeError(
            f"{name}: Ncut was asked for n_eig={n_eig}, but returned "
            f"eigvecs shape={tuple(eigvecs.shape)}. This means the backend ignored or capped n_eig."
        )
    cut_dims = min(n_clusters, int(eigvecs.shape[1]))
    z = eigvecs[:, :cut_dims].detach().float()
    print(
        f"{name}: eigvecs shape={tuple(eigvecs.shape)}, "
        f"clusterer={clusterer}, clustering_dims={cut_dims}, requested_clusters={n_clusters}"
    )

    if clusterer == "kway":
        from ncut_pytorch import kway_ncut
        # Documented API: number of output segments is controlled by the number
        # of eigenvector columns passed to kway_ncut. Do not rely on an ignored
        # or version-dependent n_clusters keyword.
        kway = kway_ncut(z)
        clusters = kway.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
    elif clusterer == "kmeans":
        z_np = z.cpu().numpy().astype(np.float32)
        try:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=10,
                batch_size=max(1024, 3 * n_clusters),
                reassignment_ratio=0.0,
            )
            clusters = km.fit_predict(z_np).astype(np.int32)
        except Exception as exc:
            print(f"{name}: MiniBatchKMeans failed ({exc}); falling back to KMeans.")
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            clusters = km.fit_predict(z_np).astype(np.int32)
    else:
        raise ValueError(f"Unknown --ncut-clusterer: {clusterer}")

    eig_np = eigvecs.detach().float().cpu().numpy()
    _print_cluster_diagnostics(name, clusters, n_clusters)
    return eig_np, clusters


def _default_mspace_n_eig_list(n_samples: int) -> List[int]:
    """Choose safe M-space supervision eig-counts for small centroid/plot sets."""
    n = max(1, int(n_samples))
    vals: List[int] = []
    for candidate in (2, 4, 8, 16, 32, 64):
        vals.append(min(candidate, n))
    return sorted(set(v for v in vals if v >= 1))


def _mspace_transform(
    points: np.ndarray,
    out_dim: int,
    seed: int,
    mspace_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Embed points with ncut_pytorch's M-space visualizer."""
    try:
        from ncut_pytorch.color.mspace import mspace_viz_transform
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "--embedder mspace needs a ncut_pytorch build that provides "
            "ncut_pytorch.color.mspace.mspace_viz_transform. Install/update from "
            "https://github.com/huzeyann/ncut_pytorch."
        ) from exc

    x_np = np.asarray(points, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"Expected 2D points for M-space, got shape {x_np.shape}")

    kwargs = dict(mspace_kwargs or {})
    kwargs.setdefault("z_dim", int(out_dim))
    kwargs.setdefault("n_eig_list", _default_mspace_n_eig_list(x_np.shape[0]))

    # M-space itself does not expose a random_state argument, so seed Torch/NumPy
    # immediately before calling it to make repeated runs as stable as possible.
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    print(
        f"Running M-space embedding: points={x_np.shape[0]:,}, "
        f"dim={x_np.shape[1]}, z_dim={kwargs.get('z_dim')}, "
        f"encoder_steps={kwargs.get('encoder_training_steps', 'default')}, "
        f"decoder_steps={kwargs.get('decoder_training_steps', 'default')}"
    )
    with torch.no_grad():
        pass
    y = mspace_viz_transform(torch.from_numpy(x_np), **kwargs)
    if torch.is_tensor(y):
        y_np = y.detach().float().cpu().numpy()
    else:
        y_np = np.asarray(y, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32)
    if y_np.ndim != 2:
        raise RuntimeError(f"M-space returned non-2D output shape {y_np.shape}")
    if y_np.shape[1] < out_dim:
        y_np = np.pad(y_np, ((0, 0), (0, out_dim - y_np.shape[1])))
    elif y_np.shape[1] > out_dim:
        y_np = y_np[:, :out_dim]
    return y_np


def embed_cluster_colors(
    eigvecs: np.ndarray,
    clusters: np.ndarray,
    method: str,
    seed: int,
    mspace_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
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
    elif method == "mspace":
        emb = _mspace_transform(centroids, out_dim=3, seed=seed, mspace_kwargs=mspace_kwargs)
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
    mspace_kwargs: dict[str, Any] | None = None,
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
    elif method == "mspace":
        # M-space learns its own affinity-preserving embedding. The feature-plot
        # metric flag is intentionally ignored for this backend.
        xy = _mspace_transform(points, out_dim=2, seed=seed, mspace_kwargs=mspace_kwargs)
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
    mspace_kwargs: dict[str, Any] | None = None,
    global_indices: np.ndarray | None = None,
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
    points = np.asarray(points, dtype=np.float32)
    if len(items) == 0:
        raise ValueError("Cannot write video-aware feature plot with zero video items.")
    if points.ndim != 2:
        raise ValueError(f"Feature plot points must be 2D, got shape {points.shape}")
    if len(points) != len(clusters):
        raise RuntimeError(
            f"Feature plot received {len(points)} points but {len(clusters)} cluster labels."
        )

    video_ids, local_token_ids, token_coords = _build_global_token_metadata(items)
    if global_indices is None:
        if len(video_ids) != len(clusters):
            raise RuntimeError(
                f"Feature plot metadata has {len(video_ids)} tokens, but clusters has {len(clusters)} labels."
            )
        point_global_indices = np.arange(len(clusters), dtype=np.int64)
    else:
        point_global_indices = np.asarray(global_indices, dtype=np.int64)
        if len(point_global_indices) != len(clusters):
            raise RuntimeError(
                f"Feature plot received {len(point_global_indices)} global indices but "
                f"{len(clusters)} cluster labels."
            )
        if len(point_global_indices) and (point_global_indices.min() < 0 or point_global_indices.max() >= len(video_ids)):
            raise RuntimeError(
                f"Feature plot global indices are out of bounds for {len(video_ids)} total tokens."
            )

    sample_rows = _stratified_sample_indices(clusters, max_points=max_points, seed=seed)
    sampled_global_indices = point_global_indices[sample_rows]
    sampled_points = np.asarray(points[sample_rows], dtype=np.float32)
    sampled_clusters = clusters[sample_rows]
    sampled_video_ids = video_ids[sampled_global_indices]

    print(
        f"Embedding {len(sample_rows):,}/{len(clusters):,} token points with "
        f"{method.upper()} for feature scatter plot..."
    )
    xy = _embed_points_2d(
        sampled_points,
        method=method,
        seed=seed,
        metric=metric,
        mspace_kwargs=mspace_kwargs,
    )
    colors = cluster_rgb[sampled_clusters]

    if write_csv:
        if out_csv is None:
            out_csv = out_png.with_name(out_png.stem + "_points.csv")
        _write_feature_plot_point_csv(
            out_csv=out_csv,
            sample_idx=sampled_global_indices,
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
        f"{len(sample_rows):,}/{len(clusters):,} tokens plotted; "
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

