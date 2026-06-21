#!/usr/bin/env bash
set -euo pipefail




# run like:

# CUDA_VISIBLE_DEVICES=0 REPO=/home/jiaray/mrBean/Synchformer \
# TRAIN_TARS=/home/jiaray/mrBean/data/webdataset_clips/train_set \
# VALID_TARS=/home/jiaray/mrBean/data/webdataset_clips/valid_set \
# TEST_TARS=/home/jiaray/mrBean/data/webdataset_clips/valid_set \
# S1_CKPT=/home/jiaray/mrBean/Synchformer/checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
# LOGDIR=/home/jiaray/mrBean/logs/synchformer_stage2_wds \
# GPU=0 \
# BATCH_SIZE=8 \
# NUM_WORKERS=8 \
# PREFETCH_FACTOR=4 \
# DECODED_CACHE_SIZE=0 \
# EXTRA_OVERRIDES="training.max_epochs=1 data.dataset.size_ratio=0.01" \
# bash /home/jiaray/mrBean/Synchformer/scripts/run_stage2_wds_finetune.sh



# Run from anywhere. Override variables on the command line, e.g.:
#   REPO=/home/jiaray/mrBean/Synchformer \
#   TRAIN_TARS=/home/jiaray/mrBean/data/train_tars \
#   VALID_TARS=/home/jiaray/mrBean/data/valid_tars \
#   S1_CKPT=/home/jiaray/mrBean/checkpoints/feature_extractors/epoch_best.pt \
#   bash run_stage2_wds_finetune.sh

REPO="${REPO:-/home/jiaray/mrBean/Synchformer}"
TRAIN_TARS="${TRAIN_TARS:-/home/jiaray/mrBean/data/webdataset_clips/train_set}"
VALID_TARS="${VALID_TARS:-/home/jiaray/mrBean/data/webdataset_clips/valid_set}"
TEST_TARS="${TEST_TARS:-$VALID_TARS}"
LOGDIR="${LOGDIR:-/home/jiaray/mrBean/logs/synchformer_stage2_wds}"

# Stage-1 AV feature-extractor checkpoint. Required for official stage-2 training.
# This is the checkpoint containing both audio and visual feature extractor weights.
S1_CKPT="${S1_CKPT:-}"

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
DECODED_CACHE_SIZE="${DECODED_CACHE_SIZE:-0}"
CACHE_DECODED="${CACHE_DECODED:-false}"
TAR_HANDLE_CACHE_SIZE="${TAR_HANDLE_CACHE_SIZE:-8}"
RECURSIVE="${RECURSIVE:-false}"
STRICT_VIDEO_FPS="${STRICT_VIDEO_FPS:-25}"
STRICT_AUDIO_FPS="${STRICT_AUDIO_FPS:-16000}"
USE_WANDB="${USE_WANDB:-false}"

if [[ -z "$S1_CKPT" ]]; then
  echo "ERROR: Set S1_CKPT=/path/to/stage1_feature_extractor_epoch_best.pt" >&2
  echo "For example: S1_CKPT=/home/jiaray/mrBean/checkpoints/23-12-22T16-10-50/checkpoints/epoch_best.pt" >&2
  exit 2
fi

if [[ ! -d "$REPO" ]]; then
  echo "ERROR: REPO does not exist: $REPO" >&2
  exit 2
fi

if [[ ! -e "$S1_CKPT" ]]; then
  echo "ERROR: S1_CKPT does not exist: $S1_CKPT" >&2
  exit 2
fi

cd "$REPO"
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python - <<'PY'
import av
print('PyAV version:', av.__version__)
PY

echo "REPO=$REPO"
echo "TRAIN_TARS=$TRAIN_TARS"
echo "VALID_TARS=$VALID_TARS"
echo "TEST_TARS=$TEST_TARS"
echo "LOGDIR=$LOGDIR"
echo "S1_CKPT=$S1_CKPT"
echo "BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS PREFETCH_FACTOR=$PREFETCH_FACTOR"
echo "DECODED_CACHE_SIZE=$DECODED_CACHE_SIZE per worker; cache is worker-local and persists across epochs only with persistent_workers=true"

python main.py \
  config=./configs/sync.yaml \
  logging.logdir="$LOGDIR" \
  data.vids_path="$TRAIN_TARS" \
  data.dataset.target=dataset.webdataset_tar_inmemory_cached_sync.WebDatasetTarInMemoryCachedSync \
  data.dataset.params.train_vids_dir="$TRAIN_TARS" \
  data.dataset.params.valid_vids_dir="$VALID_TARS" \
  data.dataset.params.test_vids_dir="$TEST_TARS" \
  data.dataset.params.recursive="$RECURSIVE" \
  data.dataset.params.cache_decoded="$CACHE_DECODED" \
  data.dataset.params.decoded_cache_size="$DECODED_CACHE_SIZE" \
  data.dataset.params.cache_tar_handles=true \
  data.dataset.params.tar_handle_cache_size="$TAR_HANDLE_CACHE_SIZE" \
  data.dataset.params.clone_cached_tensors=false \
  data.dataset.params.max_clip_len_sec=null \
  data.dataset.params.strict_video_fps="$STRICT_VIDEO_FPS" \
  data.dataset.params.strict_audio_fps="$STRICT_AUDIO_FPS" \
  model.params.vfeat_extractor.params.ckpt_path="$S1_CKPT" \
  model.params.afeat_extractor.params.ckpt_path="$S1_CKPT" \
  training.base_batch_size="$BATCH_SIZE" \
  training.num_workers="$NUM_WORKERS" \
  training.persistent_workers=true \
  training.prefetch_factor="$PREFETCH_FACTOR" \
  training.pin_memory=true \
  training.use_half_precision=true \
  training.skip_test=True \
  logging.vis_segment_sim=False \
  logging.log_code_state=False \
  logging.use_wandb="$USE_WANDB"
