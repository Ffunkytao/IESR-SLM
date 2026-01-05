#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 设置 PYTHONPATH 为项目根目录
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 默认路径配置（相对路径）
MODEL_NAME=${MODEL_NAME:-"Qwen2.5-Coder-7B-Instruct"}
GT=${GT:-"${PROJECT_ROOT}/data/logiccat/IeSlMRM_dev.json"}
PRED=${PRED:-"${PROJECT_ROOT}/results/IeSlMRM_dev/Qwen/${MODEL_NAME}/selected_sc_only.json"}
CFG=${CFG:-"${PROJECT_ROOT}/alphasql_rstar/config/logiccat_preprocess.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/results/IeSlMRM_dev/Qwen/${MODEL_NAME}/eval_outputs"}

python "${PROJECT_ROOT}/alphasql_rstar/scripts/eval_exec_acc.py" \
  --gt_data "${GT}" \
  --pred_json "${PRED}" \
  --mysql_config "${CFG}" \
  --timeout 60 \
  --output_dir "${OUTPUT_DIR}"

