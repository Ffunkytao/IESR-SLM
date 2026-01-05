#!/usr/bin/env bash
set -euo pipefail

# Archer dataset preprocessing runner for AlphaSQLFix
# - Processes SQLite databases
# - Converts archer data format to preprocessor format
# - Uses local embedding model path exported via env

# Usage:
#   bash $(dirname "$0")/run_archer_preprocess.sh [BASE_CONFIG]
# Env:
#   MAX_SAMPLES    default: 500 (set to -1 or use ALL_SAMPLES=true to process all)
#   ALL_SAMPLES    if set to true, process all samples (overrides MAX_SAMPLES)
#   EMBED_PATH     default: "" (通过环境变量设置)
#   EMBED_DEVICE   default: cuda:1
#   LLM_PATH       default: "" (通过环境变量设置)
#   LLM_DEVICE     default: cuda:0

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_CONFIG=${1:-"${PROJECT_ROOT}/alphasql_rstar/config/archer_preprocess.yaml"}

# Default values (通过环境变量设置，如果未设置则为空)
LLM_PATH=${LLM_PATH:-""}
LLM_DEVICE=${LLM_DEVICE:-cuda:0}
EMBED_PATH=${EMBED_PATH:-""}
EMBED_DEVICE=${EMBED_DEVICE:-cuda:1}

# Handle ALL_SAMPLES flag
if [[ "${ALL_SAMPLES:-false}" == "true" ]]; then
    MAX_SAMPLES=-1
    echo "[INFO] ALL_SAMPLES=true: Processing all samples"
else
    MAX_SAMPLES=${MAX_SAMPLES:-500}
fi

# Ensure Python paths
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Configure local embedding via env (picked up by alphasql_rstar.llm_call.embedding_utils)
export USE_LOCAL_EMBEDDING=true
export LOCAL_EMBEDDING_MODEL_PATH="${EMBED_PATH}"
export LOCAL_EMBEDDING_DEVICE="${EMBED_DEVICE}"

# Convert archer data format to preprocessor format
ARCHER_DATA="${PROJECT_ROOT}/data/archer/train.json"
CONVERTED_DATA=$(mktemp /tmp/archer_converted_XXXX.json)

echo "[INFO] Converting archer data format..."
python3 - "$ARCHER_DATA" "$CONVERTED_DATA" "$MAX_SAMPLES" <<'PY'
import json
import sys

input_file, output_file, max_samples = sys.argv[1], sys.argv[2], int(sys.argv[3])

with open(input_file, 'r', encoding='utf-8') as f:
    archer_data = json.load(f)

# Limit samples if needed
if max_samples != -1:
    archer_data = archer_data[:max_samples]

# Convert archer format to preprocessor format
converted_data = []
for idx, item in enumerate(archer_data):
    converted_item = {
        "question_id": idx,
        "db_id": item["db_id"],
        "question": item["question"],
        "evidence": item.get("commonsense_knowledge", ""),  # Use commonsense_knowledge as evidence
        "SQL": item["query"],  # archer uses "query" instead of "SQL"
        "difficulty": item.get("reasoning_type", None)  # Use reasoning_type as difficulty
    }
    converted_data.append(converted_item)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"Converted {len(converted_data)} samples")
PY

echo "[INFO] Converted data saved to: ${CONVERTED_DATA}"

# Create a temporary config overriding max_dataset_samples and optional LLM overrides
TMP_CFG=$(mktemp /tmp/archer_preprocess_XXXX.yaml)
python3 - "$BASE_CONFIG" "$TMP_CFG" "$MAX_SAMPLES" "${LLM_PATH}" "${LLM_DEVICE}" "$CONVERTED_DATA" <<'PY'
import sys, yaml, os
src, dst, max_samples, llm_path, llm_device, data_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
with open(src, 'r', encoding='utf-8') as f:
    d = yaml.safe_load(f)
# override only the field we need
d['max_dataset_samples'] = max_samples
d['data_file_path'] = data_file  # Use converted data file
# optional overrides for local llm
if llm_path:
    d['local_model_path'] = llm_path
if llm_device:
    d['local_model_device'] = llm_device
# ensure save dir exists
save_root_dir = d.get('save_root_dir')
if save_root_dir:
    os.makedirs(os.path.join(save_root_dir, d.get('split', 'train')), exist_ok=True)
with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)
print(dst)
PY

echo "[INFO] Using base config: ${BASE_CONFIG}"
echo "[INFO] Effective config: ${TMP_CFG} (max_dataset_samples=${MAX_SAMPLES})"
echo "[INFO] Embedding model: ${LOCAL_EMBEDDING_MODEL_PATH} on ${LOCAL_EMBEDDING_DEVICE}"
echo "[INFO] Local LLM path: ${LLM_PATH}"
echo "[INFO] Local LLM device: ${LLM_DEVICE}"

# Update config with converted data path
python3 - "$TMP_CFG" "$CONVERTED_DATA" <<'PY'
import sys, yaml
cfg_file, data_file = sys.argv[1], sys.argv[2]
with open(cfg_file, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg['data_file_path'] = data_file
with open(cfg_file, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

# Run the preprocessor using YAML config wrapper
python3 -m alphasql_rstar.runner.preprocessor_archer "${TMP_CFG}"

# Cleanup
rm -f "${TMP_CFG}" "${CONVERTED_DATA}"
echo "[INFO] Preprocessing complete!"

