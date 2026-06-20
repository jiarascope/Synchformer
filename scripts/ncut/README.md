# Refactored NCut video script

This folder is a split version of the original `ncut_video.py`.

Run it the same way, from your repo root, for example:

```bash
python3 ncut_video.py \
  --repo_root . \
  --checkpoint checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
  --video /path/to/video.mp4 \
  --out_dir outputs/ncut_umap50 \
  --whole_video \
  --segment_sec 0.64 \
  --stride_sec 0.32 \
  --max_duration_sec 5 \
  --num_frames 16 \
  --image_size 224 \
  --patch_size 16 \
  --num_eig 50 \
  --eig_rgb_dims 50 \
  --num_clusters 10 \
  --alpha 0.25 \
  --device cuda
```

If want to run on directory of videos:

'''bash

CUDA_VISIBLE_DEVICES=0 python3 scripts/ncut/ncut_video.py \
  --repo_root . \
  --checkpoint checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
  --video_dir /home/jiaray/mrBean/data/ncut_smalltest \
  --out_dir outputs/joint_ncut \
  --joint_video_dir_ncut \
  --segment_sec 0.64 \
  --stride_sec 0.32 \
  --max_duration_sec 5 \
  --num_frames 16 \
  --image_size 224 \
  --patch_size 16 \
  --num_eig 50 \
  --eig_rgb_dims 50 \
  --num_clusters 6 \
  --alpha 0.55 \
  --device cuda
'''

If you keep this inside `scripts/ncut/`, leave your existing `preprocess.py` next to `ncut_video.py`.

## Files

- `ncut_video.py`: small runnable entry point and argument parsing.
- `ncut_modules/model.py`: Synchformer/MotionFormer loading and token extraction.
- `ncut_modules/ncut_ops.py`: NCut wrapper.
- `ncut_modules/viz.py`: visualization, overlays, videos, contact sheets, plots.
- `ncut_modules/video_io.py`: video discovery and basic metadata.
- `ncut_modules/process_clip.py`: single clip processing.
- `ncut_modules/process_whole.py`: whole-video sliding window processing.
- `ncut_modules/process_joint.py`: joint NCut over a video directory.



