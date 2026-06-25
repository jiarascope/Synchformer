# Refactored NCut video script

This folder is a split version of the original `ncut_video.py`.

Run it from your repo root. `ncut_video.py` now runs one joint NCut over all
videos in an input directory and writes only mp4 overlay videos to `--out_dir`.

```bash
python3 ncut_video.py \
  --repo_root /home/jiaray/mrBean/Synchformer \
  --checkpoint checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
  --video_dir /home/jiaray/mrBean/data/baseline_data/conducting_clips \
  --out_dir /home/jiaray/mrBean/plots/baseline \
  --embedding_map umap \
  --segment_sec 0.64 \
  --stride_sec 0.32 \
  --max_duration_sec 5 \
  --num_frames 16 \
  --image_size 224 \
  --patch_size 16 \
  --num_eig 50 \
  --eig_rgb_dims 50 \
  --num_clusters 30 \
  --alpha 0.55 \
  --device cuda
```

Use `--embedding_map tsne` instead of `umap` to make the continuous RGB overlay
from a 3D t-SNE projection. The cluster overlay always uses the shared k-means
labels from the joint NCut embedding.

For each input video, `--out_dir` gets:

- `<video_stem>_joint_<umap|tsne>_rgb_overlay.mp4`
- `<video_stem>_joint_clusters_overlay.mp4`

If you keep this inside `scripts/ncut/`, leave your existing `preprocess.py` next to `ncut_video.py`.

## Files

- `ncut_video.py`: runnable joint-directory NCut entry point and argument parsing.
- `ncut_modules/model.py`: Synchformer/MotionFormer loading and token extraction.
- `ncut_modules/ncut_ops.py`: NCut wrapper.
- `ncut_modules/viz.py`: visualization, overlays, videos, contact sheets, plots.
- `ncut_modules/video_io.py`: video discovery and basic metadata.
- `ncut_modules/process_clip.py`: single clip processing.
- `ncut_modules/process_whole.py`: whole-video sliding window processing.
- `ncut_modules/process_joint.py`: joint NCut over a video directory.


