#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_entropy_grid_common.sh"

GRID_LIMIT=100
GRID_MAX_NEW_TOKENS=256
GRID_STEPS=128
GRID_BLOCK_SIZE=32
GRID_CREDIT_RATES=(0.20 0.30 0.40 0.50)
GRID_WARMUP_RATIOS=(0.00 0.05)
GRID_ACTIVE_END_RATIOS=(0.10 0.20)
GRID_END_RATIOS=(0.25 0.30 0.35)
SWEEP_NAME_SUFFIX="gsm8k-cot-limit100-len256-step128-entropy-grid48-gpu2"
EXPECTED_COMBINATION_COUNT=48

ensure_environment
require_model_path
build_all_combinations
resolve_sweep_root
initialize_sweep_root

run_grid_slice 2 0 47
