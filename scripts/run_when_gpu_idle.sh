#!/usr/bin/env bash
# Wait until a GPU has no active compute processes, then start a target command.
#
# Example:
#   export MODEL_PATH=/disk/wangzhe/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
#   bash /disk/wangzhe/dllm/scripts/run_when_gpu_idle.sh \
#     --gpu 0 \
#     --poll-seconds 60 \
#     -- bash /disk/wangzhe/dllm/scripts/run_entropy_grid_gpu0.sh

set -euo pipefail

GPU_INDEX=""
POLL_SECONDS=60
LOG_FILE=""

usage() {
  cat <<'EOF'
Usage:
  bash run_when_gpu_idle.sh --gpu <gpu_index> [--poll-seconds <seconds>] [--log-file <path>] -- <command ...>

Options:
  --gpu             GPU index to monitor, e.g. 0
  --poll-seconds    Poll interval in seconds, default 60
  --log-file        Optional log file for wait/start messages
EOF
}

log_message() {
  local message="$1"
  local timestamp
  timestamp="$(date --iso-8601=seconds)"
  printf '[%s] %s\n' "${timestamp}" "${message}"
  if [[ -n "${LOG_FILE}" ]]; then
    printf '[%s] %s\n' "${timestamp}" "${message}" >> "${LOG_FILE}"
  fi
}

gpu_uuid_for_index() {
  local gpu_index="$1"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | \
    awk -F', ' -v target="${gpu_index}" '$1 == target { print $2; exit }'
}

gpu_busy_process_count() {
  local gpu_uuid="$1"
  local app_rows
  app_rows="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${app_rows}" ]]; then
    echo 0
    return
  fi
  printf '%s\n' "${app_rows}" | awk -F', ' -v target="${gpu_uuid}" '$1 == target { count += 1 } END { print count + 0 }'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_INDEX="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${GPU_INDEX}" ]]; then
  echo "--gpu is required." >&2
  usage >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Target command is required after --." >&2
  usage >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Please run this on a GPU server with NVIDIA tooling installed." >&2
  exit 1
fi

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
fi

GPU_UUID="$(gpu_uuid_for_index "${GPU_INDEX}")"
if [[ -z "${GPU_UUID}" ]]; then
  echo "Unable to resolve GPU UUID for GPU index ${GPU_INDEX}." >&2
  exit 1
fi

log_message "Watching GPU ${GPU_INDEX} (${GPU_UUID}) with poll interval ${POLL_SECONDS}s."

while true; do
  BUSY_COUNT="$(gpu_busy_process_count "${GPU_UUID}")"
  if [[ "${BUSY_COUNT}" -eq 0 ]]; then
    log_message "GPU ${GPU_INDEX} is idle. Starting command: $*"
    exec "$@"
  fi

  log_message "GPU ${GPU_INDEX} still busy with ${BUSY_COUNT} compute process(es). Sleeping ${POLL_SECONDS}s."
  sleep "${POLL_SECONDS}"
done
