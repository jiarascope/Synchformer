#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere. Override variables on the command line, e.g.:
#   REPO=/home/jiaray/mrBean/Synchformer \
#   TRAIN_TARS=/home/jiaray/mrBean/data/train_tars \
#   VALID_TARS=/home/jiaray/mrBean/data/valid_tars \
#   S1_CKPT=/home/jiaray/mrBean/Synchformer/checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
#   NUM_EPOCHS=5 \
#   LOG_MAX_ITEMS=512 \
#   LOG_FREQUENCY=100 \
#   bash run_stage2_wds_finetune.sh


# CUDA_VISIBLE_DEVICES=0 S1_CKPT=/home/jiaray/mrBean/Synchformer/checkpoints/segment_avclip/synchformer_avclip_audioset.pt BATCH_SIZE=4 NUM_WORKERS=2 PREFETCH_FACTOR=1 NUM_EPOCHS=5 CACHE_DECODED=false CACHE_TAR_HANDLES=false DEBUG_IO=false PERSISTENT_WORKERS=true LOG_FREQUENCY=500 LOG_MAX_ITEMS=32 USE_WANDB=false bash /home/jiaray/mrBean/Synchformer/scripts/run_stage2_wds_finetune.sh training.resume=True ckpt_path=/home/jiaray/mrBean/logs/synchformer_stage2_wds/26-06-22T20-31-00/26-06-22T20-31-00_latest.pt start_time=26-06-22T20-31-00

REPO="${REPO:-/home/jiaray/mrBean/Synchformer}"
TRAIN_TARS="${TRAIN_TARS:-/home/jiaray/mrBean/data/webdataset_clips/train_set}"
VALID_TARS="${VALID_TARS:-/home/jiaray/mrBean/data/webdataset_clips/valid_set}"
TEST_TARS="${TEST_TARS:-$VALID_TARS}"
LOGDIR="${LOGDIR:-/home/jiaray/mrBean/logs/synchformer_stage2_wds}"

# Stage-1 AV feature-extractor checkpoint. Required for official stage-2 training.
# This is the checkpoint containing both audio and visual feature extractor weights.
S1_CKPT="${S1_CKPT:-}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DECODED_CACHE_SIZE="${DECODED_CACHE_SIZE:-0}"
CACHE_DECODED="${CACHE_DECODED:-false}"
CACHE_TAR_HANDLES="${CACHE_TAR_HANDLES:-false}"
TAR_HANDLE_CACHE_SIZE="${TAR_HANDLE_CACHE_SIZE:-8}"
RECURSIVE="${RECURSIVE:-false}"
STRICT_VIDEO_FPS="${STRICT_VIDEO_FPS:-25}"
STRICT_AUDIO_FPS="${STRICT_AUDIO_FPS:-16000}"
USE_WANDB="${USE_WANDB:-false}"

NUM_EPOCHS="${NUM_EPOCHS:-5}"
LOG_FREQUENCY="${LOG_FREQUENCY:-100}"
LOG_MAX_ITEMS="${LOG_MAX_ITEMS:-512}"

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
echo "GPU=$GPU"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "PREFETCH_FACTOR=$PREFETCH_FACTOR"
echo "NUM_EPOCHS=$NUM_EPOCHS"
echo "LOG_FREQUENCY=$LOG_FREQUENCY"
echo "LOG_MAX_ITEMS=$LOG_MAX_ITEMS"
echo "CACHE_DECODED=$CACHE_DECODED"
echo "DECODED_CACHE_SIZE=$DECODED_CACHE_SIZE per worker; cache is worker-local and persists across epochs only with persistent_workers=true"
echo "TAR_HANDLE_CACHE_SIZE=$TAR_HANDLE_CACHE_SIZE"
echo "CACHE_TAR_HANDLES=$CACHE_TAR_HANDLES"

set -x

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
  data.dataset.params.cache_tar_handles="$CACHE_TAR_HANDLES" \
  data.dataset.params.tar_handle_cache_size="$TAR_HANDLE_CACHE_SIZE" \
  data.dataset.params.clone_cached_tensors=false \
  data.dataset.params.max_clip_len_sec=null \
  data.dataset.params.strict_video_fps="$STRICT_VIDEO_FPS" \
  data.dataset.params.strict_audio_fps="$STRICT_AUDIO_FPS" \
  model.params.vfeat_extractor.params.ckpt_path="$S1_CKPT" \
  model.params.afeat_extractor.params.ckpt_path="$S1_CKPT" \
  training.base_batch_size="$BATCH_SIZE" \
  training.num_workers="$NUM_WORKERS" \
  training.num_epochs="$NUM_EPOCHS" \
  training.persistent_workers=true \
  training.prefetch_factor="$PREFETCH_FACTOR" \
  training.pin_memory=false \
  data.dataset.params.debug_io="${DEBUG_IO:-false}" \
  data.dataset.params.worker_threads=1 \
  data.dataset.params.decode_threads=1 \
  training.use_half_precision=true \
  training.skip_test=True \
  logging.log_frequency="$LOG_FREQUENCY" \
  logging.log_max_items="$LOG_MAX_ITEMS" \
  logging.vis_segment_sim=False \
  logging.log_code_state=False \
  logging.use_wandb="$USE_WANDB"\
  "$@"

set +x