#!/bin/bash
# 启动本地 vLLM 服务器
# 
# 用法:
#   bash run_local_model.sh [MODEL_PATH] [MODEL_NAME] [PORT] [TP_SIZE]
#
# 环境变量:
#   MODEL_PATH      模型路径（默认: 通过环境变量设置或使用参数）
#   MODEL_NAME      模型名称（默认: Qwen/Qwen2.5-Coder-7B-Instruct）
#   PORT            服务端口（默认: 9999）
#   TP_SIZE         张量并行大小（默认: 4）
#   CUDA_VISIBLE_DEVICES 可见的 GPU 设备（默认: 0,1,2,3）
#   LOG_FILE        日志文件路径（默认: logs/vllm_9999.log）

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 创建日志目录
mkdir -p "${PROJECT_ROOT}/logs"

# 参数设置（支持命令行参数和环境变量）
MODEL_PATH=${1:-${MODEL_PATH:-""}}
MODEL_NAME=${2:-${MODEL_NAME:-"Qwen/Qwen2.5-Coder-7B-Instruct"}}
PORT=${3:-${PORT:-9999}}
TP_SIZE=${4:-${TP_SIZE:-4}}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
LOG_FILE=${LOG_FILE:-"${PROJECT_ROOT}/logs/vllm_${PORT}.log"}

# 检查模型路径
if [ -z "$MODEL_PATH" ]; then
    echo "错误: 未指定模型路径"
    echo "请通过以下方式之一设置:"
    echo "  1. 命令行参数: bash run_local_model.sh /path/to/model"
    echo "  2. 环境变量: export MODEL_PATH=/path/to/model"
    exit 1
fi

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 模型路径不存在: $MODEL_PATH"
    echo "请确认模型路径是否正确"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 提取模型短名称用于日志
MODEL_SHORT_NAME=$(basename "$MODEL_PATH")

echo "=========================================="
echo "启动 vLLM 服务器"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "服务端口: $PORT"
echo "张量并行: $TP_SIZE"
echo "GPU 设备: $CUDA_DEVICES"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 启动 vLLM 服务器
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
stdbuf -oL -eL python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization 0.80 \
  --max-model-len 16384 \
  2>&1 | tee -a "$LOG_FILE"