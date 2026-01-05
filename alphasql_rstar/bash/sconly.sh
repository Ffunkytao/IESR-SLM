#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 设置 PYTHONPATH 为项目根目录
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 默认使用 MCTS 配置中的保存目录
MODEL_NAME=${MODEL_NAME:-"Qwen2.5-Coder-7B-Instruct"}
RESULTS_DIR=${1:-"${PROJECT_ROOT}/results/IeSlMRM_dev/Qwen/${MODEL_NAME}"}
OUTPUT_PATH=${2:-"${RESULTS_DIR}/selected_sc_only.json"}

python "${PROJECT_ROOT}/alphasql_rstar/scripts/select_sc_only.py" \
  --results_dir "${RESULTS_DIR}" \
  --output_path "${OUTPUT_PATH}"

