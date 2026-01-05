#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 设置 PYTHONPATH 为项目根目录
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 模型配置（可通过命令行参数覆盖）
MODEL_NAME=${1:-"Qwen/Qwen2.5-Coder-7B-Instruct"}

# OpenAI API 配置（用于 vLLM 服务器）
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-"http://0.0.0.0:9999/v1"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-"dummy"}

# SQL 执行配置
# mysql is 0 sqlite is 1
export ALPHASQL_SKIP_SQL_EXEC=${ALPHASQL_SKIP_SQL_EXEC:-0}
export ALPHASQL_SQL_BACKEND=${ALPHASQL_SQL_BACKEND:-mysql}

# MySQL 连接配置（从环境变量读取，如果未设置则使用默认值）
# 注意：请通过环境变量设置数据库连接信息，不要硬编码
export MYSQL_HOST=${MYSQL_HOST:-""}
export MYSQL_PORT=${MYSQL_PORT:-3306}
export MYSQL_USER=${MYSQL_USER:-"root"}
export MYSQL_PASSWORD=${MYSQL_PASSWORD:-""}

# 验证执行开关执行情况 0 open 1 close
export ALPHASQL_LOG_SQL=${ALPHASQL_LOG_SQL:-0}
# 动态空间开关执行情况 1 open 0 close
export ALPHASQL_ACTION_LOG_SQL=${ALPHASQL_ACTION_LOG_SQL:-1}

# 配置文件路径（相对路径）
CONFIG_FILE="${SCRIPT_DIR}/../config/mcts_logiccat.yaml"

# 运行 MCTS runner，传递模型名称
python -m alphasql_rstar.runner.mcts_runner \
  "${CONFIG_FILE}" \
  --model "${MODEL_NAME}"

