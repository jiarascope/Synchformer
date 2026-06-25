#!/usr/bin/env bash
set -euo pipefail

# Standalone LoRA stage-1 fine-tuning for Synchformer AVCLIP on MP4-in-tar shards.
# This does NOT patch the Synchformer repo. It calls train_stage1_lora_wds.py, which
# imports Synchformer modules normally and saves a merged epoch_best.pt / epoch_latest.pt
# that can be used as S1_CKPT for your existing stage-2 script.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 \
#   S1_CKPT=/home/jiaray/mrBean/Synchformer/checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
#   BATCH_SIZE=1 NUM_WORKERS=8 PREFETCH_FACTOR=2 NUM_EPOCHS=5 \
#   bash /home/jiaray/mrBean/Synchformer/scripts/run_stage1_lora_wds.sh

# CUDA_VISIBLE_DEVICES=0 \
# S1_CKPT=/home/jiaray/mrBean/Synchformer/checkpoints/segment_avclip/synchformer_avclip_audioset.pt \
# BATCH_SIZE=1 \
# NUM_WORKERS=8 \
# PREFETCH_FACTOR=2 \
# NUM_EPOCHS=3 \
# LORA_RANK=8 \
# LORA_ALPHA=16 \
# LORA_TARGET_MODE=attention \
# bash /home/jiaray/mrBean/Synchformer/scripts/run_stage1_lora_wds.sh

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] [run_stage1_lora_wds] %s\n' -1 "$*"
}

warn() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] [run_stage1_lora_wds] WARN: %s\n' -1 "$*" >&2
}

die() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] [run_stage1_lora_wds] ERROR: %s\n' -1 "$*" >&2
  exit 2
}

count_tars() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    printf 'missing'
    return
  fi
  find "$path" -type f \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) 2>/dev/null | wc -l
}

print_kv() {
  printf '  %-28s %s\n' "$1" "$2"
}


REPO="${REPO:-/home/jiaray/mrBean/Synchformer}"
LORA_SCRIPT="${LORA_SCRIPT:-$REPO/scripts/train_stage1_lora_wds.py}"
CONFIG="${CONFIG:-$REPO/configs/segment_avclip.yaml}"

TRAIN_TARS="${TRAIN_TARS:-/home/jiaray/mrBean/data/webdataset_clips/train_set}"
VALID_TARS="${VALID_TARS:-/home/jiaray/mrBean/data/webdataset_clips/valid_set}"
TEST_TARS="${TEST_TARS:-$VALID_TARS}"
LOGDIR="${LOGDIR:-/home/jiaray/mrBean/logs/synchformer_stage1_lora_wds}"

# Existing stage-1 AVCLIP checkpoint to adapt. Required.
S1_CKPT="${S1_CKPT:-}"
RESUME_LORA="${RESUME_LORA:-}"
ALLOW_NONSTRICT_S1_LOAD="${ALLOW_NONSTRICT_S1_LOAD:-false}"

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-false}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
N_SEGMENTS="${N_SEGMENTS:-14}"
FOR_LOOP_SEGMENT_FWD="${FOR_LOOP_SEGMENT_FWD:-true}"
RUN_SHIFTED_WIN_VAL="${RUN_SHIFTED_WIN_VAL:-true}"
VAL_FREQUENCY="${VAL_FREQUENCY:-1}"

# LoRA knobs. Use attention first; attention_mlp gives more adaptation capacity.
LORA_SCOPE="${LORA_SCOPE:-both}"                  # both | audio | visual
LORA_TARGET_MODE="${LORA_TARGET_MODE:-attention}" # attention | attention_mlp | all_linear
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TRAIN_LOGIT_SCALE="${TRAIN_LOGIT_SCALE:-true}"
TRAIN_LAYER_NORM="${TRAIN_LAYER_NORM:-false}"
TRAIN_BIAS="${TRAIN_BIAS:-false}"
TRAIN_PROJ="${TRAIN_PROJ:-false}"

# Dataset/decoder knobs. Stage-1 transform itself caps clips at 10s; decoding only 10s saves RAM/CPU
# if your tar members are longer. Set MAX_CLIP_LEN_SEC=null to decode full members.
MAX_CLIP_LEN_SEC="${MAX_CLIP_LEN_SEC:-10}"
CACHE_DECODED="${CACHE_DECODED:-false}"
DECODED_CACHE_SIZE="${DECODED_CACHE_SIZE:-0}"
CACHE_TAR_HANDLES="${CACHE_TAR_HANDLES:-false}"
TAR_HANDLE_CACHE_SIZE="${TAR_HANDLE_CACHE_SIZE:-8}"
RECURSIVE="${RECURSIVE:-false}"
STRICT_VIDEO_FPS="${STRICT_VIDEO_FPS:-25}"
STRICT_AUDIO_FPS="${STRICT_AUDIO_FPS:-16000}"
DEBUG_IO="${DEBUG_IO:-false}"
PROFILE_IO="${PROFILE_IO:-false}"

LOG_FREQUENCY="${LOG_FREQUENCY:-100}"
LOG_MAX_ITEMS="${LOG_MAX_ITEMS:-128}"
USE_WANDB="${USE_WANDB:-false}"
DEBUG_SHELL="${DEBUG_SHELL:-false}"
PRINT_ENV="${PRINT_ENV:-false}"

mkdir -p "$LOGDIR"
RUN_LOG="$LOGDIR/run_stage1_lora_wds_$(date +%Y%m%d_%H%M%S).log"
TEE_FIFO="$(mktemp -u "${TMPDIR:-/tmp}/run_stage1_lora_wds.XXXXXX")"
mkfifo "$TEE_FIFO"
exec 3>&1 4>&2
tee -a "$RUN_LOG" < "$TEE_FIFO" >&3 &
TEE_PID=$!
exec > "$TEE_FIFO" 2>&1
rm -f "$TEE_FIFO"

cleanup_tee() {
  local status=$?
  exec 1>&3 2>&4
  exec 3>&- 4>&-
  wait "$TEE_PID" 2>/dev/null || true
  exit "$status"
}
trap cleanup_tee EXIT

if [[ "$DEBUG_SHELL" == "true" ]]; then
  PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
  set -x
fi

log "Wrapper log: $RUN_LOG"

if [[ -z "$S1_CKPT" ]]; then
  die "Set S1_CKPT=/path/to/existing/stage1_avclip_checkpoint.pt"
fi
if [[ ! -d "$REPO" ]]; then
  die "REPO does not exist: $REPO"
fi
if [[ ! -f "$LORA_SCRIPT" ]]; then
  die "LORA_SCRIPT does not exist: $LORA_SCRIPT. Put train_stage1_lora_wds.py at $REPO/scripts/train_stage1_lora_wds.py or set LORA_SCRIPT=/path/to/it"
fi
if [[ ! -f "$CONFIG" ]]; then
  die "CONFIG does not exist: $CONFIG"
fi
if [[ ! -e "$S1_CKPT" ]]; then
  die "S1_CKPT does not exist: $S1_CKPT"
fi

cd "$REPO"
log "Working directory: $(pwd)"
log "Validating paths and dataset inputs"
if [[ ! -d "$TRAIN_TARS" ]]; then warn "TRAIN_TARS directory does not exist: $TRAIN_TARS"; fi
if [[ ! -d "$VALID_TARS" ]]; then warn "VALID_TARS directory does not exist: $VALID_TARS"; fi
if [[ ! -d "$TEST_TARS" ]]; then warn "TEST_TARS directory does not exist: $TEST_TARS"; fi

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-1}"

log "Checking Python/PyAV import"
python - <<'PY'
import av
print('PyAV version:', av.__version__)
PY

log "Run configuration"
print_kv "REPO" "$REPO"
print_kv "LORA_SCRIPT" "$LORA_SCRIPT"
print_kv "CONFIG" "$CONFIG"
print_kv "TRAIN_TARS" "$TRAIN_TARS"
print_kv "VALID_TARS" "$VALID_TARS"
print_kv "TEST_TARS" "$TEST_TARS"
print_kv "TRAIN_TAR_COUNT" "$(count_tars "$TRAIN_TARS")"
print_kv "VALID_TAR_COUNT" "$(count_tars "$VALID_TARS")"
print_kv "TEST_TAR_COUNT" "$(count_tars "$TEST_TARS")"
print_kv "LOGDIR" "$LOGDIR"
print_kv "RUN_LOG" "$RUN_LOG"
print_kv "S1_CKPT" "$S1_CKPT"
print_kv "RESUME_LORA" "${RESUME_LORA:-<none>}"
print_kv "ALLOW_NONSTRICT_S1_LOAD" "$ALLOW_NONSTRICT_S1_LOAD"
print_kv "CUDA_VISIBLE_DEVICES" "$CUDA_VISIBLE_DEVICES"
print_kv "BATCH_SIZE" "$BATCH_SIZE"
print_kv "NUM_WORKERS" "$NUM_WORKERS"
print_kv "PREFETCH_FACTOR" "$PREFETCH_FACTOR"
print_kv "PERSISTENT_WORKERS" "$PERSISTENT_WORKERS"
print_kv "PIN_MEMORY" "$PIN_MEMORY"
print_kv "LEARNING_RATE" "$LEARNING_RATE"
print_kv "NUM_EPOCHS" "$NUM_EPOCHS"
print_kv "N_SEGMENTS" "$N_SEGMENTS"
print_kv "FOR_LOOP_SEGMENT_FWD" "$FOR_LOOP_SEGMENT_FWD"
print_kv "RUN_SHIFTED_WIN_VAL" "$RUN_SHIFTED_WIN_VAL"
print_kv "VAL_FREQUENCY" "$VAL_FREQUENCY"
print_kv "LORA_SCOPE" "$LORA_SCOPE"
print_kv "LORA_TARGET_MODE" "$LORA_TARGET_MODE"
print_kv "LORA_RANK" "$LORA_RANK"
print_kv "LORA_ALPHA" "$LORA_ALPHA"
print_kv "LORA_DROPOUT" "$LORA_DROPOUT"
print_kv "TRAIN_LOGIT_SCALE" "$TRAIN_LOGIT_SCALE"
print_kv "TRAIN_LAYER_NORM" "$TRAIN_LAYER_NORM"
print_kv "TRAIN_BIAS" "$TRAIN_BIAS"
print_kv "TRAIN_PROJ" "$TRAIN_PROJ"
print_kv "MAX_CLIP_LEN_SEC" "$MAX_CLIP_LEN_SEC"
print_kv "CACHE_DECODED" "$CACHE_DECODED"
print_kv "DECODED_CACHE_SIZE" "$DECODED_CACHE_SIZE"
print_kv "CACHE_TAR_HANDLES" "$CACHE_TAR_HANDLES"
print_kv "TAR_HANDLE_CACHE_SIZE" "$TAR_HANDLE_CACHE_SIZE"
print_kv "RECURSIVE" "$RECURSIVE"
print_kv "STRICT_VIDEO_FPS" "$STRICT_VIDEO_FPS"
print_kv "STRICT_AUDIO_FPS" "$STRICT_AUDIO_FPS"
print_kv "DEBUG_IO" "$DEBUG_IO"
print_kv "PROFILE_IO" "$PROFILE_IO"
print_kv "LOG_FREQUENCY" "$LOG_FREQUENCY"
print_kv "LOG_MAX_ITEMS" "$LOG_MAX_ITEMS"
print_kv "USE_WANDB" "$USE_WANDB"
print_kv "DEBUG_SHELL" "$DEBUG_SHELL"

if [[ "$PRINT_ENV" == "true" ]]; then
  log "Environment snapshot"
  env | sort
fi

PY_ARGS=(
  "$LORA_SCRIPT"
  --repo "$REPO"
  --config "$CONFIG"
  --s1-ckpt "$S1_CKPT"
  --lora-scope "$LORA_SCOPE"
  --lora-target-mode "$LORA_TARGET_MODE"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
)

if [[ "$TRAIN_LOGIT_SCALE" == "true" ]]; then PY_ARGS+=(--train-logit-scale); else PY_ARGS+=(--no-train-logit-scale); fi
if [[ "$TRAIN_LAYER_NORM" == "true" ]]; then PY_ARGS+=(--train-layer-norm); else PY_ARGS+=(--no-train-layer-norm); fi
if [[ "$TRAIN_BIAS" == "true" ]]; then PY_ARGS+=(--train-bias); else PY_ARGS+=(--no-train-bias); fi
if [[ "$TRAIN_PROJ" == "true" ]]; then PY_ARGS+=(--train-proj); else PY_ARGS+=(--no-train-proj); fi
if [[ -n "$RESUME_LORA" ]]; then PY_ARGS+=(--resume-lora "$RESUME_LORA"); fi
if [[ "$ALLOW_NONSTRICT_S1_LOAD" == "true" ]]; then PY_ARGS+=(--allow-nonstrict-s1-load); fi

log "Launching LoRA training"
printf '  python'
printf ' %q' "${PY_ARGS[@]}" --
printf ' %q' \
  logging.logdir="$LOGDIR" \
  logging.log_frequency="$LOG_FREQUENCY" \
  logging.log_max_items="$LOG_MAX_ITEMS" \
  logging.log_code_state=false \
  logging.use_wandb="$USE_WANDB" \
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
  data.dataset.params.max_clip_len_sec="$MAX_CLIP_LEN_SEC" \
  data.dataset.params.strict_video_fps="$STRICT_VIDEO_FPS" \
  data.dataset.params.strict_audio_fps="$STRICT_AUDIO_FPS" \
  data.dataset.params.debug_io="$DEBUG_IO" \
  data.dataset.params.profile_io="$PROFILE_IO" \
  data.dataset.params.worker_threads=1 \
  data.dataset.params.decode_threads=1 \
  training.base_batch_size="$BATCH_SIZE" \
  training.num_workers="$NUM_WORKERS" \
  training.num_epochs="$NUM_EPOCHS" \
  training.learning_rate="$LEARNING_RATE" \
  training.persistent_workers="$PERSISTENT_WORKERS" \
  training.prefetch_factor="$PREFETCH_FACTOR" \
  training.pin_memory="$PIN_MEMORY" \
  training.precision=amp \
  training.for_loop_segment_fwd="$FOR_LOOP_SEGMENT_FWD" \
  training.run_shifted_win_val="$RUN_SHIFTED_WIN_VAL" \
  training.val_frequency="$VAL_FREQUENCY" \
  data.n_segments_train="$N_SEGMENTS" \
  data.n_segments_valid="$N_SEGMENTS" \
  "$@"
printf '\n'

set +e
python "${PY_ARGS[@]}" -- \
  logging.logdir="$LOGDIR" \
  logging.log_frequency="$LOG_FREQUENCY" \
  logging.log_max_items="$LOG_MAX_ITEMS" \
  logging.log_code_state=false \
  logging.use_wandb="$USE_WANDB" \
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
  data.dataset.params.max_clip_len_sec="$MAX_CLIP_LEN_SEC" \
  data.dataset.params.strict_video_fps="$STRICT_VIDEO_FPS" \
  data.dataset.params.strict_audio_fps="$STRICT_AUDIO_FPS" \
  data.dataset.params.debug_io="$DEBUG_IO" \
  data.dataset.params.profile_io="$PROFILE_IO" \
  data.dataset.params.worker_threads=1 \
  data.dataset.params.decode_threads=1 \
  training.base_batch_size="$BATCH_SIZE" \
  training.num_workers="$NUM_WORKERS" \
  training.num_epochs="$NUM_EPOCHS" \
  training.learning_rate="$LEARNING_RATE" \
  training.persistent_workers="$PERSISTENT_WORKERS" \
  training.prefetch_factor="$PREFETCH_FACTOR" \
  training.pin_memory="$PIN_MEMORY" \
  training.precision=amp \
  training.for_loop_segment_fwd="$FOR_LOOP_SEGMENT_FWD" \
  training.run_shifted_win_val="$RUN_SHIFTED_WIN_VAL" \
  training.val_frequency="$VAL_FREQUENCY" \
  data.n_segments_train="$N_SEGMENTS" \
  data.n_segments_valid="$N_SEGMENTS" \
  "$@"
status=$?
set -e
log "Training command exited with status $status"
log "Wrapper log saved to: $RUN_LOG"
exit "$status"
