from __future__ import annotations

from torch.utils.data import Dataset, Subset

try:
    from dataset.webdataset_tar_inmemory import WebDatasetTarInMemoryCachedSync
except ImportError as exc:
    raise ImportError(
        "Could not import dataset.webdataset_tar_inmemory_fast.WebDatasetTarInMemoryCachedSync.\n"
        "Copy dataset/webdataset_tar_inmemory_fast.py from this package into the Synchformer "
        "repo's dataset/ directory."
    ) from exc


class RepeatedTrialDataset(Dataset):
    """
    Flatten stochastic trials into the sample index.

    Ordering is intentionally clip-major:
        clip0 trial0, clip0 trial1, ..., clip1 trial0, ...

    That makes repeated trials of the same clip adjacent, so the cached dataset
    can reuse decoded rgb/audio tensors before moving to the next clip.
    """

    def __init__(self, base_dataset: Dataset, data_iter: int):
        self.base_dataset = base_dataset
        self.data_iter = int(data_iter)
        if self.data_iter <= 0:
            raise RuntimeError(f"data_iter must be positive, got {data_iter}")

    def __len__(self) -> int:
        return len(self.base_dataset) * self.data_iter

    def __getitem__(self, index: int):
        base_index = index // self.data_iter
        iter_idx = index % self.data_iter

        item = self.base_dataset[base_index]
        item["iter_idx"] = int(iter_idx)
        return item


def build_eval_dataset(args, transform):
    """Build cached base dataset, apply optional max_samples, then flatten trials."""
    base_dataset = WebDatasetTarInMemoryCachedSync(
        split="test",
        vids_dir=args.tar_dir,
        transforms=transform,
        load_fixed_offsets_on=[],
        recursive=args.recursive,
        cache_decoded=args.cache_decoded,
        decoded_cache_size=args.decoded_cache_size,
        cache_tar_handles=args.cache_tar_handles,
        tar_handle_cache_size=args.tar_handle_cache_size,
        clone_cached_tensors=args.clone_cached_tensors,
        strict_video_fps=args.strict_video_fps,
        strict_audio_fps=args.strict_audio_fps,
        max_clip_len_sec=args.max_clip_len_sec,
    )

    if args.max_samples is not None:
        max_samples = min(int(args.max_samples), len(base_dataset))
        base_dataset = Subset(base_dataset, list(range(max_samples)))
        num_clips = max_samples
    else:
        num_clips = len(base_dataset)

    repeated_dataset = RepeatedTrialDataset(base_dataset, args.data_iter)
    return repeated_dataset, num_clips
