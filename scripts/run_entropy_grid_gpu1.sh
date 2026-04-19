#!/usr/bin/env bash
set -euo pipefail

source /disk/wangzhe/dllm/scripts/run_entropy_grid_common.sh

ensure_environment
require_model_path
build_all_combinations
resolve_sweep_root
initialize_sweep_root

run_grid_slice 1 36 71
