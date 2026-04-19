#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_entropy_grid_common.sh"

ensure_environment
require_model_path
build_all_combinations
resolve_sweep_root
initialize_sweep_root

run_grid_slice 0 0 35
