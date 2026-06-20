# Fast modular Synchformer batch eval

This package replaces the single `batch_eval.py` script with a modular version and a faster tar dataset adapter.

## Files

Copy these into the root of your Synchformer repo:

```text
batch_eval_fast.py
fast_batch_eval/
dataset/webdataset_tar_inmemory_fast.py
```

`dataset/webdataset_tar_inmemory_fast.py` is intentionally separate from your existing
`dataset/webdataset_tar_inmemory.py`, so you can keep the old adapter around.

## What is faster

The old script did:

```text
for iter in data_iter:
  for clip in dataset:
    open tar -> read mp4 -> decode video/audio -> crop -> GPU forward
```

The new script does:

```text
for clip/trial in clip-major flattened order:
  clip0 trial0, clip0 trial1, ..., clip1 trial0, ...
```

and the new dataset adapter caches decoded raw `rgb` and `audio` tensors per DataLoader worker.

It also:
- caches open tar handles per worker
- computes softmax/top-k on GPU
- uses `torch.inference_mode()`
- supports `prefetch_factor`
- batches CSV row writes per batch

## Example command

```bash
CUDA_VISIBLE_DEVICES=0 python3 batch_eval.py   --tar_dir /home/jiaray/mrBean/data/baseline_data/tarfiles   --device cuda:0   --batch_size 32   --num_workers 8   --persistent_workers   --prefetch_factor 4   --data_iter 5   --decoded_cache_size 64   --out_csv tables/stochastic_trial_predictions1.csv
```

## Tuning

Watch `nvidia-smi`.

- GPU memory low and GPU not saturated: increase `--batch_size`.
- GPU waits on CPU: increase `--num_workers` or `--prefetch_factor`.
- RAM is available: increase `--decoded_cache_size`.
- Transform mutates tensors in place or outputs look suspicious: add `--clone_cached_tensors`.
- Want to test without cache: add `--no_cache_decoded`.

`--decoded_cache_size 0` means unlimited per worker. Be careful: raw decoded video tensors are large.
