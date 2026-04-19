#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GRID_TASK="gsm8k_cot"
GRID_NUM_FEWSHOT=5
GRID_LIMIT=200
GRID_MODEL="llada"
GRID_MAX_NEW_TOKENS=256
GRID_STEPS=64
GRID_BLOCK_SIZE=32
GRID_CFG_SCALE=0.0
GRID_BEGIN_SUPPRESS_TOKENS="[126081;126348]"
GRID_ENTROPY_TOP_K=64

GRID_CREDIT_RATES=(0.20 0.30 0.40)
GRID_WARMUP_RATIOS=(0.00 0.05)
GRID_ACTIVE_END_RATIOS=(0.10 0.15 0.20)
GRID_END_RATIOS=(0.25 0.30 0.35 0.40)

SWEEP_BASE_ROOT="${REPO_ROOT}/.logs/sweeps"
SWEEP_REGISTRY_PATH="${SWEEP_BASE_ROOT}/latest-gsm8k-cot-limit200-entropy-grid72.path"
SWEEP_REGISTRY_LOCK="${SWEEP_BASE_ROOT}/.latest-gsm8k-cot-limit200-entropy-grid72.lock"

ensure_environment() {
  if [[ -f "${HOME}/.zshrc" ]]; then
    source "${HOME}/.zshrc"
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" != "dllm" ]]; then
    local conda_candidate
    local conda_loaded=0
    for conda_candidate in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/miniforge3/etc/profile.d/conda.sh" \
      "/disk/wangzhe/miniconda3/etc/profile.d/conda.sh" \
      "/home/wangzhe/miniconda3/etc/profile.d/conda.sh"; do
      if [[ -f "${conda_candidate}" ]]; then
        source "${conda_candidate}"
        conda_loaded=1
        break
      fi
    done

    if [[ "${conda_loaded}" -ne 1 ]]; then
      echo "Unable to locate conda.sh. Please initialize conda before running this script." >&2
      exit 1
    fi

    conda activate dllm
  fi

  cd "${REPO_ROOT}"
  export PYTHONPATH=.:$PYTHONPATH
}

require_model_path() {
  if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "MODEL_PATH is required. Export MODEL_PATH before running this script." >&2
    exit 1
  fi
}

float_slug() {
  local value="$1"
  printf '%s' "${value//./p}"
}

build_all_combinations() {
  GRID_COMBINATIONS=()
  local credit_rate warmup_ratio active_end_ratio end_ratio
  for credit_rate in "${GRID_CREDIT_RATES[@]}"; do
    for warmup_ratio in "${GRID_WARMUP_RATIOS[@]}"; do
      for active_end_ratio in "${GRID_ACTIVE_END_RATIOS[@]}"; do
        for end_ratio in "${GRID_END_RATIOS[@]}"; do
          GRID_COMBINATIONS+=(
            "${credit_rate}|${warmup_ratio}|${active_end_ratio}|${end_ratio}"
          )
        done
      done
    done
  done

  if [[ "${#GRID_COMBINATIONS[@]}" -ne 72 ]]; then
    echo "Expected 72 combinations, got ${#GRID_COMBINATIONS[@]}." >&2
    exit 1
  fi
}

resolve_sweep_root() {
  mkdir -p "${SWEEP_BASE_ROOT}"

  if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    SWEEP_ROOT="${OUTPUT_ROOT}"
    mkdir -p "${SWEEP_ROOT}"
    printf '%s\n' "${SWEEP_ROOT}" > "${SWEEP_REGISTRY_PATH}"
    return
  fi

  exec 201>"${SWEEP_REGISTRY_LOCK}"
  flock 201

  if [[ -f "${SWEEP_REGISTRY_PATH}" ]]; then
    local existing_root
    existing_root="$(<"${SWEEP_REGISTRY_PATH}")"
    if [[ -n "${existing_root}" && -d "${existing_root}" ]]; then
      local existing_runs_root="${existing_root}/runs"
      local done_count=0
      if [[ -d "${existing_runs_root}" ]]; then
        done_count="$(find "${existing_runs_root}" -name done.ok | wc -l | tr -d ' ')"
      fi

      if [[ "${done_count}" -lt 72 ]]; then
        SWEEP_ROOT="${existing_root}"
      fi
    fi
  fi

  if [[ -z "${SWEEP_ROOT:-}" ]]; then
    local timestamp
    timestamp="$(date +%Y%m%d-%H%M%S)"
    SWEEP_ROOT="${SWEEP_BASE_ROOT}/${timestamp}-gsm8k-cot-limit200-entropy-grid72"
    mkdir -p "${SWEEP_ROOT}"
    printf '%s\n' "${SWEEP_ROOT}" > "${SWEEP_REGISTRY_PATH}"
  fi
}

initialize_sweep_root() {
  mkdir -p "${SWEEP_ROOT}/runs"

  if [[ ! -f "${SWEEP_ROOT}/sweep_config.json" ]]; then
    cat > "${SWEEP_ROOT}/sweep_config.json" <<EOF
{
  "task": "${GRID_TASK}",
  "num_fewshot": ${GRID_NUM_FEWSHOT},
  "limit": ${GRID_LIMIT},
  "model": "${GRID_MODEL}",
  "max_new_tokens": ${GRID_MAX_NEW_TOKENS},
  "steps": ${GRID_STEPS},
  "block_size": ${GRID_BLOCK_SIZE},
  "cfg_scale": ${GRID_CFG_SCALE},
  "begin_suppress_tokens": "${GRID_BEGIN_SUPPRESS_TOKENS}",
  "entropy_top_k": ${GRID_ENTROPY_TOP_K},
  "grid": {
    "entropy_credit_rate": [0.20, 0.30, 0.40],
    "entropy_warmup_ratio": [0.00, 0.05],
    "entropy_active_end_ratio": [0.10, 0.15, 0.20],
    "entropy_end_ratio": [0.25, 0.30, 0.35, 0.40]
  }
}
EOF
  fi
}

write_meta_json() {
  local run_dir="$1"
  local run_id="$2"
  local slug="$3"
  local assigned_gpu="$4"
  local credit_rate="$5"
  local warmup_ratio="$6"
  local active_end_ratio="$7"
  local end_ratio="$8"
  local status="$9"
  local exit_code="${10}"
  local started_at="${11}"
  local finished_at="${12}"
  local eval_log_path="${13}"
  local generation_records_path="${14}"

  cat > "${run_dir}/meta.json" <<EOF
{
  "run_id": "${run_id}",
  "slug": "${slug}",
  "assigned_gpu": ${assigned_gpu},
  "status": "${status}",
  "exit_code": ${exit_code},
  "started_at": "${started_at}",
  "finished_at": "${finished_at}",
  "eval_log_path": "${eval_log_path}",
  "generation_records_path": "${generation_records_path}",
  "params": {
    "entropy_credit_rate": ${credit_rate},
    "entropy_warmup_ratio": ${warmup_ratio},
    "entropy_active_end_ratio": ${active_end_ratio},
    "entropy_end_ratio": ${end_ratio},
    "entropy_top_k": ${GRID_ENTROPY_TOP_K}
  },
  "metrics": {
    "flexible_exact_match": null,
    "strict_exact_match": null,
    "flexible_stderr": null,
    "strict_stderr": null,
    "duration": null
  },
  "note": ""
}
EOF
}

write_command_script() {
  local command_path="$1"
  shift
  local -a command=("$@")

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "${command_path}"
  chmod +x "${command_path}"
}

update_reports() {
  local sweep_root="$1"
  {
    flock 202
    python "${REPO_ROOT}/scripts/update_entropy_grid_reports.py" \
      --sweep-root "${sweep_root}"
  } 202>"${sweep_root}/.report.lock"
}

run_grid_slice() {
  local assigned_gpu="$1"
  local start_index="$2"
  local end_index="$3"

  local offset combination credit_rate warmup_ratio active_end_ratio end_ratio
  for ((offset = start_index; offset <= end_index; offset++)); do
    IFS='|' read -r credit_rate warmup_ratio active_end_ratio end_ratio <<< "${GRID_COMBINATIONS[$offset]}"

    local run_number=$((offset + 1))
    local run_id
    run_id="$(printf 'r%03d' "${run_number}")"
    local slug
    slug="cr$(float_slug "${credit_rate}")-wu$(float_slug "${warmup_ratio}")-ae$(float_slug "${active_end_ratio}")-ee$(float_slug "${end_ratio}")"
    local run_dir="${SWEEP_ROOT}/runs/${run_id}__${slug}"
    local eval_log_path="${run_dir}/eval.log"
    local generation_records_path="${run_dir}/generation_records.jsonl"
    local command_path="${run_dir}/command.sh"
    local done_marker="${run_dir}/done.ok"
    local failed_marker="${run_dir}/failed.ok"

    if [[ -f "${done_marker}" ]]; then
      echo "[GPU ${assigned_gpu}] Skip ${run_id} (${slug}) because done.ok exists."
      continue
    fi

    mkdir -p "${run_dir}"
    rm -f "${done_marker}" "${failed_marker}"

    local model_args="pretrained=${MODEL_PATH},max_new_tokens=${GRID_MAX_NEW_TOKENS},steps=${GRID_STEPS},block_size=${GRID_BLOCK_SIZE},cfg_scale=${GRID_CFG_SCALE},suppress_tokens=[],begin_suppress_tokens=${GRID_BEGIN_SUPPRESS_TOKENS},enable_entropy_priority=True,enable_entropy_credit_scheduler=True,entropy_credit_rate=${credit_rate},entropy_warmup_ratio=${warmup_ratio},entropy_active_end_ratio=${active_end_ratio},entropy_end_ratio=${end_ratio},entropy_top_k=${GRID_ENTROPY_TOP_K},save_generation_records_path=${generation_records_path},save_generation_traces=True,generation_trace_max_steps=64,save_sampler_diagnostics=True,diagnostic_collect_step_debug=True,diagnostic_log_interval=1"
    local -a command=(
      accelerate launch --num_processes 1 "${REPO_ROOT}/dllm/pipelines/llada/eval.py"
      --tasks "${GRID_TASK}"
      --num_fewshot "${GRID_NUM_FEWSHOT}"
      --limit "${GRID_LIMIT}"
      --model "${GRID_MODEL}"
      --apply_chat_template
      --model_args "${model_args}"
    )

    write_command_script "${command_path}" "CUDA_VISIBLE_DEVICES=${assigned_gpu}" "${command[@]}"

    local started_at
    started_at="$(date --iso-8601=seconds)"
    write_meta_json \
      "${run_dir}" \
      "${run_id}" \
      "${slug}" \
      "${assigned_gpu}" \
      "${credit_rate}" \
      "${warmup_ratio}" \
      "${active_end_ratio}" \
      "${end_ratio}" \
      "running" \
      -1 \
      "${started_at}" \
      "" \
      "${eval_log_path}" \
      "${generation_records_path}"

    echo "[GPU ${assigned_gpu}] Start ${run_id} (${slug})"
    local exit_code=0
    CUDA_VISIBLE_DEVICES="${assigned_gpu}" "${command[@]}" 2>&1 | tee "${eval_log_path}" || exit_code=$?

    local finished_at
    finished_at="$(date --iso-8601=seconds)"
    if [[ "${exit_code}" -eq 0 ]]; then
      touch "${done_marker}"
      write_meta_json \
        "${run_dir}" \
        "${run_id}" \
        "${slug}" \
        "${assigned_gpu}" \
        "${credit_rate}" \
        "${warmup_ratio}" \
        "${active_end_ratio}" \
        "${end_ratio}" \
        "completed" \
        "${exit_code}" \
        "${started_at}" \
        "${finished_at}" \
        "${eval_log_path}" \
        "${generation_records_path}"
      echo "[GPU ${assigned_gpu}] Completed ${run_id} (${slug})"
    else
      touch "${failed_marker}"
      write_meta_json \
        "${run_dir}" \
        "${run_id}" \
        "${slug}" \
        "${assigned_gpu}" \
        "${credit_rate}" \
        "${warmup_ratio}" \
        "${active_end_ratio}" \
        "${end_ratio}" \
        "failed" \
        "${exit_code}" \
        "${started_at}" \
        "${finished_at}" \
        "${eval_log_path}" \
        "${generation_records_path}"
      echo "[GPU ${assigned_gpu}] Failed ${run_id} (${slug}) with exit code ${exit_code}" >&2
    fi

    update_reports "${SWEEP_ROOT}"
  done
}
