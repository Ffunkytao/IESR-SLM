# IESR - Intelligent SQL Reasoning with MCTS

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**基于蒙特卡洛树搜索（MCTS）的智能 SQL 推理系统**

[English](#english) | [中文](#中文)

</div>

---

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

IESR 是一个基于蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）算法的智能 SQL 生成与推理系统。该系统能够处理复杂的自然语言查询，并生成高质量的 SQL 语句，特别适用于需要复杂推理的 SQL 生成任务。

### 核心特性

- 🎯 **MCTS 算法**：使用蒙特卡洛树搜索进行 SQL 生成空间探索
- 🧠 **大语言模型集成**：支持多种 LLM（默认使用 Qwen2.5-Coder-7B-Instruct）
- 🔍 **智能检索**：基于 LSH 和嵌入向量的相似 SQL 检索
- 📊 **多数据集支持**：支持 LogicCat 和 Archer 数据集
- 🗄️ **多数据库支持**：支持 MySQL 和 SQLite

---

## 功能特性

- ✅ **灵活的模型配置**：支持通过命令行参数指定模型
- ✅ **相对路径支持**：所有路径配置使用相对路径，便于复现
- ✅ **环境变量配置**：数据库连接等敏感信息通过环境变量管理
- ✅ **并行处理**：支持多进程并行处理任务
- ✅ **结果评估**：内置 SQL 执行准确率评估工具

---

## 环境要求

### 系统要求

- **操作系统**：Linux / macOS / Windows (WSL)
- **Python**：3.8 或更高版本
- **GPU**：推荐使用 NVIDIA GPU（用于本地模型推理）

### Python 依赖

主要依赖包包括：

```bash
# 核心依赖
pydantic>=2.0.0
pyyaml>=6.0
loguru>=0.7.0
openai>=1.0.0
python-dotenv>=1.0.0

# 数据库相关
pymysql>=1.0.0
sqlalchemy>=2.0.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0

# 其他
tqdm>=4.65.0
```

**vLLM 依赖（如果使用本地模型服务器）：**

```bash
# 安装 vLLM（需要 CUDA 支持）
pip install vllm
```

> **注意**：完整的依赖列表请参考项目中的 `requirements.txt`（如果存在）或根据运行时的错误提示安装相应包。

---

## 快速开始

### 1. 克隆仓库

```bash
git clone <repository-url>
cd IESR
```

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt  # 如果有 requirements.txt
# 或手动安装主要依赖
pip install pydantic pyyaml loguru openai python-dotenv pymysql sqlalchemy numpy pandas tqdm
```

### 3. 配置环境变量

创建 `.env` 文件（可选，部分配置可通过环境变量设置）：

```bash
# OpenAI API 配置（用于 vLLM 服务器）
export OPENAI_BASE_URL="http://0.0.0.0:9999/v1"
export OPENAI_API_KEY="dummy"

# MySQL 数据库配置（如果使用 MySQL）
export MYSQL_HOST="your_mysql_host"
export MYSQL_PORT="3306"
export MYSQL_USER="root"
export MYSQL_PASSWORD="your_password"

# SQL 执行配置
export ALPHASQL_SKIP_SQL_EXEC=0  # 0: 执行 SQL, 1: 跳过执行
export ALPHASQL_SQL_BACKEND=mysql  # mysql 或 sqlite

# 本地模型配置（如果使用本地模型）
export LOCAL_EMBEDDING_MODEL_PATH="/path/to/embedding/model"
export USE_LOCAL_EMBEDDING=true
```

### 4. 准备数据

确保数据文件位于正确的位置：

```bash
# LogicCat 数据集
data/logiccat/IeSlMRM_dev.json

# Archer 数据集
data/archer/train.json
```

### 5. 启动本地模型服务器（可选）

如果你使用本地模型通过 vLLM 提供服务，需要先启动服务器：

```bash
# 方式 1: 使用命令行参数
bash alphasql_rstar/bash/run_local_model.sh \
  /path/to/your/model \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  9999 \
  4

# 方式 2: 使用环境变量
export MODEL_PATH="/path/to/your/model"
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export PORT=9999
export TP_SIZE=4
bash alphasql_rstar/bash/run_local_model.sh

# 方式 3: 混合使用（参数 + 环境变量）
export MODEL_PATH="/path/to/your/model"
bash alphasql_rstar/bash/run_local_model.sh "" "Qwen/Qwen2.5-Coder-7B-Instruct" 9999 4
```

**参数说明：**
- `MODEL_PATH`: 本地模型路径（必需）
- `MODEL_NAME`: 模型名称，用于 API 调用（默认: `Qwen/Qwen2.5-Coder-7B-Instruct`）
- `PORT`: 服务端口（默认: `9999`）
- `TP_SIZE`: 张量并行大小，根据 GPU 数量设置（默认: `4`）
- `CUDA_VISIBLE_DEVICES`: 可见的 GPU 设备（默认: `0,1,2,3`）

**注意：** 服务器启动后，日志会保存在 `logs/vllm_<PORT>.log` 文件中。

### 6. 运行示例

```bash
# 进入项目根目录
cd ~/IESR

# 如果使用本地 vLLM 服务器，确保服务器已启动
# 然后运行 MCTS 推理（使用默认模型）
bash alphasql_rstar/bash/run_mcts.sh

# 或指定模型
bash alphasql_rstar/bash/run_mcts.sh "Qwen/Qwen2.5-Coder-7B-Instruct"
```

---

## 使用指南

### 启动本地模型服务器

#### 使用 vLLM 启动本地模型

如果你有本地模型文件，可以使用 vLLM 启动一个 OpenAI 兼容的 API 服务器：

```bash
# 基本用法
bash alphasql_rstar/bash/run_local_model.sh /path/to/model

# 完整参数
bash alphasql_rstar/bash/run_local_model.sh \
  /path/to/model \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  9999 \
  4
```

**参数说明：**
- 第一个参数：模型路径（必需）
- 第二个参数：模型名称（默认: `Qwen/Qwen2.5-Coder-7B-Instruct`）
- 第三个参数：服务端口（默认: `9999`）
- 第四个参数：张量并行大小（默认: `4`）

**环境变量配置：**

```bash
# 通过环境变量设置
export MODEL_PATH="/path/to/your/model"
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export PORT=9999
export TP_SIZE=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
bash alphasql_rstar/bash/run_local_model.sh
```

**配置 vLLM 参数：**

如果需要调整 vLLM 的参数（如 GPU 内存利用率、最大序列长度等），可以直接编辑 `run_local_model.sh` 脚本。

**验证服务器：**

服务器启动后，可以通过以下方式验证：

```bash
# 检查服务是否运行
curl http://localhost:9999/v1/models

# 或查看日志
tail -f logs/vllm_9999.log
```

### 数据预处理

#### LogicCat 数据集预处理

```bash
bash alphasql_rstar/bash/run_preprocess.sh
```

或指定配置文件：

```bash
bash alphasql_rstar/bash/run_preprocess.sh alphasql_rstar/config/logiccat_preprocess.yaml
```

#### Archer 数据集预处理

```bash
bash alphasql_rstar/bash/run_archer_preprocess.sh
```

### MCTS 推理

#### 基本用法

```bash
# 使用默认模型（Qwen2.5-Coder-7B-Instruct）
bash alphasql_rstar/bash/run_mcts.sh

# 指定模型
bash alphasql_rstar/bash/run_mcts.sh "Qwen/Qwen2.5-Coder-7B-Instruct"

# 或直接使用 Python 模块
python -m alphasql_rstar.runner.mcts_runner \
  alphasql_rstar/config/mcts_logiccat.yaml \
  --model "Qwen/Qwen2.5-Coder-7B-Instruct"
```

#### 自定义配置

编辑 `alphasql_rstar/config/mcts_logiccat.yaml` 文件以调整 MCTS 参数：

```yaml
n_processes: 32              # 并行进程数
max_rollout_steps: 8         # 最大 rollout 步数
max_depth: 8                 # 最大搜索深度
exploration_constant: 1.4    # 探索常数
```

### 结果处理

#### 选择最佳 SQL

```bash
# 从 MCTS 结果中选择最佳 SQL
bash alphasql_rstar/bash/sconly.sh

# 或指定结果目录
bash alphasql_rstar/bash/sconly.sh results/IeSlMRM_dev/Qwen/Qwen2.5-Coder-7B-Instruct
```

#### 评估执行准确率

```bash
# 评估 SQL 执行准确率
bash alphasql_rstar/bash/eval.sh

# 或指定路径
export GT="data/logiccat/IeSlMRM_dev.json"
export PRED="results/IeSlMRM_dev/Qwen/Qwen2.5-Coder-7B-Instruct/selected_sc_only.json"
bash alphasql_rstar/bash/eval.sh
```

---

## 配置说明

### MCTS 配置文件

主要配置文件：`alphasql_rstar/config/mcts_logiccat.yaml`

```yaml
tasks_file_path: "data/preprocessed/IeSlMRM_dev/dev/tasks.pkl"
db_root_dir: "data/preprocessed/IeSlMRM_dev"
n_processes: 32
max_rollout_steps: 8
max_depth: 8
exploration_constant: 1.4
save_root_dir: "results/IeSlMRM_dev/Qwen/Qwen2.5-Coder-7B-Instruct"

mcts_model_kwargs:
  model: "Qwen/Qwen2.5-Coder-7B-Instruct"  # 可通过命令行参数覆盖
  temperature: 0.7
  top_p: 0.8
  max_tokens: 2048
```

### 预处理配置文件

#### LogicCat 预处理配置

`alphasql_rstar/config/logiccat_preprocess.yaml`

```yaml
data_file_path: "data/logiccat/IeSlMRM_dev.json"
save_root_dir: "data/preprocessed/IeSlMRM_dev"
lsh_threshold: 0.5
embedding_similarity_threshold: 0.6
n_parallel_processes: 8
```

#### Archer 预处理配置

`alphasql_rstar/config/archer_preprocess.yaml`

```yaml
data_file_path: "data/archer/train.json"
database_root_dir: "data/archer/database"
save_root_dir: "data/preprocessed/archer"
```

---

## 项目结构

```
IESR/
├── alphasql_rstar/          # 核心代码
│   ├── algorithm/           # MCTS 算法实现
│   │   └── mcts/            # MCTS 相关模块
│   ├── bash/                # 运行脚本
│   │   ├── run_mcts.sh      # MCTS 推理脚本
│   │   ├── run_local_model.sh # 启动本地 vLLM 服务器脚本
│   │   ├── run_preprocess.sh # 预处理脚本
│   │   ├── eval.sh          # 评估脚本
│   │   └── sconly.sh        # SQL 选择脚本
│   ├── config/              # 配置文件
│   │   ├── mcts_logiccat.yaml
│   │   ├── logiccat_preprocess.yaml
│   │   └── archer_preprocess.yaml
│   ├── database/            # 数据库相关模块
│   ├── llm_call/            # LLM 调用模块
│   ├── runner/              # 运行器模块
│   └── scripts/             # 工具脚本
├── data/                    # 数据目录
│   ├── logiccat/            # LogicCat 数据集
│   └── archer/              # Archer 数据集
├── logs/                    # 日志目录（vLLM 服务器日志）
├── results/                 # 结果目录
└── README.md               # 本文件
```

---

## 常见问题

### Q1: 如何指定不同的模型？

**A:** 有两种方式：

1. **通过命令行参数**（推荐）：
   ```bash
   bash alphasql_rstar/bash/run_mcts.sh "Qwen/Qwen2.5-Coder-7B-Instruct"
   ```

2. **修改配置文件**：
   编辑 `alphasql_rstar/config/mcts_logiccat.yaml` 中的 `mcts_model_kwargs.model` 字段

### Q2: 如何配置数据库连接？

**A:** 通过环境变量设置：

```bash
export MYSQL_HOST="your_host"
export MYSQL_PORT="3306"
export MYSQL_USER="root"
export MYSQL_PASSWORD="your_password"
```

### Q3: 如何跳过 SQL 执行验证？

**A:** 设置环境变量：

```bash
export ALPHASQL_SKIP_SQL_EXEC=1
```

### Q4: 如何处理路径问题？

**A:** 所有脚本已改为使用相对路径。确保在项目根目录（`~/IESR`）下运行脚本：

```bash
cd ~/IESR
bash alphasql_rstar/bash/run_mcts.sh
```

### Q5: 如何调整并行进程数？

**A:** 修改配置文件中的 `n_processes` 参数：

```yaml
# alphasql_rstar/config/mcts_logiccat.yaml
n_processes: 16  # 根据你的 CPU 核心数调整
```

### Q6: 如何启动本地模型服务器？

**A:** 使用 `run_local_model.sh` 脚本：

```bash
# 指定模型路径
bash alphasql_rstar/bash/run_local_model.sh /path/to/your/model

# 或使用环境变量
export MODEL_PATH="/path/to/your/model"
bash alphasql_rstar/bash/run_local_model.sh
```

### Q7: 如何调整 vLLM 服务器的 GPU 使用？

**A:** 通过 `CUDA_VISIBLE_DEVICES` 环境变量设置：

```bash
# 只使用 GPU 0 和 1
export CUDA_VISIBLE_DEVICES="0,1"
export TP_SIZE=2  # 张量并行大小需要与 GPU 数量匹配
bash alphasql_rstar/bash/run_local_model.sh /path/to/model
```

---

## 贡献指南

我们欢迎所有形式的贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 Python 代码规范
- 添加必要的注释和文档字符串
- 确保代码通过基本的语法检查

---

## 许可证

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。

---

## 致谢

感谢所有为本项目做出贡献的开发者和研究人员。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送 Pull Request

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！⭐**

</div>

---

## English

### Quick Start (English)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd IESR
   ```

2. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install pydantic pyyaml loguru openai python-dotenv pymysql sqlalchemy numpy pandas tqdm
   
   # If using local vLLM server
   pip install vllm
   ```

3. **Set environment variables**
   ```bash
   export OPENAI_BASE_URL="http://0.0.0.0:9999/v1"
   export OPENAI_API_KEY="dummy"
   export MYSQL_HOST="your_mysql_host"
   export MYSQL_PORT="3306"
   export MYSQL_USER="root"
   export MYSQL_PASSWORD="your_password"
   ```

4. **Start local model server (optional)**
   ```bash
   # Start vLLM server with local model
   bash alphasql_rstar/bash/run_local_model.sh /path/to/your/model
   
   # Or with full parameters
   bash alphasql_rstar/bash/run_local_model.sh \
     /path/to/model \
     "Qwen/Qwen2.5-Coder-7B-Instruct" \
     9999 \
     4
   ```

5. **Run MCTS inference**
   ```bash
   bash alphasql_rstar/bash/run_mcts.sh "Qwen/Qwen2.5-Coder-7B-Instruct"
   ```

### Local Model Server Usage

The `run_local_model.sh` script helps you start a local vLLM server:

```bash
# Basic usage
bash alphasql_rstar/bash/run_local_model.sh /path/to/model

# With environment variables
export MODEL_PATH="/path/to/your/model"
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export PORT=9999
export TP_SIZE=4
bash alphasql_rstar/bash/run_local_model.sh
```

**Parameters:**
- `MODEL_PATH`: Local model path (required)
- `MODEL_NAME`: Model name for API calls (default: `Qwen/Qwen2.5-Coder-7B-Instruct`)
- `PORT`: Service port (default: `9999`)
- `TP_SIZE`: Tensor parallel size (default: `4`)
- `CUDA_VISIBLE_DEVICES`: Visible GPU devices (default: `0,1,2,3`)

For more details, please refer to the Chinese documentation above.

