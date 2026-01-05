# IESR - Intelligent SQL Reasoning with MCTS

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Efficient MCTS-Based Modular Reasoning for Text-to-SQL with Small Language Models**

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

IESR (Intelligent SQL Reasoning) is an efficient Monte Carlo Tree Search (MCTS)-based modular reasoning system designed specifically for Text-to-SQL tasks. The system employs a modular reasoning approach that decomposes complex SQL generation into manageable steps, enabling effective SQL generation and reasoning with small language models (SLMs) while maintaining high accuracy and significantly reducing computational costs.

### Key Innovations

- 🎯 **Efficient MCTS Algorithm**: Optimized Monte Carlo Tree Search strategy that efficiently explores the SQL generation space, balancing exploration and exploitation
- 🧠 **Small Language Model Support**: Specifically designed for small language models (e.g., Qwen2.5-Coder-7B-Instruct), reducing deployment costs while maintaining high performance
- 🔧 **Modular Reasoning Architecture**: Decomposes complex SQL generation tasks into modular steps, improving reasoning efficiency and interpretability
- 🔍 **Intelligent Retrieval Mechanism**: Similar SQL retrieval based on Locality-Sensitive Hashing (LSH) and embedding vectors, accelerating candidate SQL generation
- 📊 **Multi-Dataset Support**: Supports complex reasoning datasets such as LogicCat and Archer, validating system performance in complex scenarios
- 🗄️ **Multi-Database Support**: Supports various database systems including MySQL and SQLite

---

## Key Features

- ✅ **Flexible Model Configuration**: Support for specifying models via command-line arguments
- ✅ **Relative Path Support**: All path configurations use relative paths for easy reproduction
- ✅ **Environment Variable Configuration**: Sensitive information such as database connections managed through environment variables
- ✅ **Parallel Processing**: Support for multi-process parallel task processing
- ✅ **Result Evaluation**: Built-in SQL execution accuracy evaluation tools
- ✅ **Modular Design**: Clear separation of concerns with modular components for preprocessing, reasoning, and evaluation

---

## System Requirements

### Operating System

- **OS**: Linux / macOS / Windows (WSL)
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU recommended (for local model inference)

### Python Dependencies

Main dependency packages include:

```bash
# Core dependencies
pydantic>=2.0.0
pyyaml>=6.0
loguru>=0.7.0
openai>=1.0.0
python-dotenv>=1.0.0

# Database related
pymysql>=1.0.0
sqlalchemy>=2.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# Other
tqdm>=4.65.0
```

**vLLM Dependencies (if using local model server):**

```bash
# Install vLLM (requires CUDA support)
pip install vllm
```

> **Note**: For the complete dependency list, please refer to `requirements.txt` in the project (if available) or install packages based on runtime error messages.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd IESR
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt  # if requirements.txt exists
# or manually install main dependencies
pip install pydantic pyyaml loguru openai python-dotenv pymysql sqlalchemy numpy pandas tqdm
```

### 3. Configure Environment Variables

Create a `.env` file (optional, some configurations can be set via environment variables):

```bash
# OpenAI API configuration (for vLLM server)
export OPENAI_BASE_URL="http://0.0.0.0:9999/v1"
export OPENAI_API_KEY="dummy"

# MySQL database configuration (if using MySQL)
export MYSQL_HOST="your_mysql_host"
export MYSQL_PORT="3306"
export MYSQL_USER="root"
export MYSQL_PASSWORD="your_password"

# SQL execution configuration
export ALPHASQL_SKIP_SQL_EXEC=0  # 0: execute SQL, 1: skip execution
export ALPHASQL_SQL_BACKEND=mysql  # mysql or sqlite

# Local model configuration (if using local models)
export LOCAL_EMBEDDING_MODEL_PATH="/path/to/embedding/model"
export USE_LOCAL_EMBEDDING=true
```

### 4. Prepare Data

Ensure data files are located in the correct positions:

```bash
# LogicCat dataset
data/logiccat/IeSlMRM_dev.json

# Archer dataset
data/archer/train.json
```

### 5. Start Local Model Server (Optional)

If you are using a local model served via vLLM, start the server first:

```bash
# Method 1: Using command-line arguments
bash alphasql_rstar/bash/run_local_model.sh \
  /path/to/your/model \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  9999 \
  4

# Method 2: Using environment variables
export MODEL_PATH="/path/to/your/model"
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export PORT=9999
export TP_SIZE=4
bash alphasql_rstar/bash/run_local_model.sh

# Method 3: Mixed usage (arguments + environment variables)
export MODEL_PATH="/path/to/your/model"
bash alphasql_rstar/bash/run_local_model.sh "" "Qwen/Qwen2.5-Coder-7B-Instruct" 9999 4
```

**Parameter Description:**
- `MODEL_PATH`: Local model path (required)
- `MODEL_NAME`: Model name for API calls (default: `Qwen/Qwen2.5-Coder-7B-Instruct`)
- `PORT`: Service port (default: `9999`)
- `TP_SIZE`: Tensor parallel size, set according to GPU count (default: `4`)
- `CUDA_VISIBLE_DEVICES`: Visible GPU devices (default: `0,1,2,3`)

**Note:** After the server starts, logs will be saved in the `logs/vllm_<PORT>.log` file.

### 6. Run Example

```bash
# Navigate to project root directory
cd ~/IESR

# If using local vLLM server, ensure the server is running
# Then run MCTS inference (using default model)
bash alphasql_rstar/bash/run_mcts.sh

# Or specify a model
bash alphasql_rstar/bash/run_mcts.sh "Qwen/Qwen2.5-Coder-7B-Instruct"
```

---

## Usage Guide

### Starting Local Model Server

#### Using vLLM to Start Local Model

If you have local model files, you can use vLLM to start an OpenAI-compatible API server:

```bash
# Basic usage
bash alphasql_rstar/bash/run_local_model.sh /path/to/model

# Full parameters
bash alphasql_rstar/bash/run_local_model.sh \
  /path/to/model \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  9999 \
  4
```

**Parameter Description:**
- First parameter: Model path (required)
- Second parameter: Model name (default: `Qwen/Qwen2.5-Coder-7B-Instruct`)
- Third parameter: Service port (default: `9999`)
- Fourth parameter: Tensor parallel size (default: `4`)

**Environment Variable Configuration:**

```bash
# Set via environment variables
export MODEL_PATH="/path/to/your/model"
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export PORT=9999
export TP_SIZE=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
bash alphasql_rstar/bash/run_local_model.sh
```

**Configuring vLLM Parameters:**

If you need to adjust vLLM parameters (such as GPU memory utilization, maximum sequence length, etc.), you can directly edit the `run_local_model.sh` script.

**Verifying Server:**

After the server starts, you can verify it using:

```bash
# Check if service is running
curl http://localhost:9999/v1/models

# Or view logs
tail -f logs/vllm_9999.log
```

### Data Preprocessing

#### LogicCat Dataset Preprocessing

```bash
bash alphasql_rstar/bash/run_preprocess.sh
```

Or specify a configuration file:

```bash
bash alphasql_rstar/bash/run_preprocess.sh alphasql_rstar/config/logiccat_preprocess.yaml
```

#### Archer Dataset Preprocessing

```bash
bash alphasql_rstar/bash/run_archer_preprocess.sh
```

### MCTS Inference

#### Basic Usage

```bash
# Use default model (Qwen2.5-Coder-7B-Instruct)
bash alphasql_rstar/bash/run_mcts.sh

# Specify model
bash alphasql_rstar/bash/run_mcts.sh "Qwen/Qwen2.5-Coder-7B-Instruct"

# Or use Python module directly
python -m alphasql_rstar.runner.mcts_runner \
  alphasql_rstar/config/mcts_logiccat.yaml \
  --model "Qwen/Qwen2.5-Coder-7B-Instruct"
```

#### Custom Configuration

Edit the `alphasql_rstar/config/mcts_logiccat.yaml` file to adjust MCTS parameters:

```yaml
n_processes: 32              # Number of parallel processes
max_rollout_steps: 8         # Maximum rollout steps
max_depth: 8                 # Maximum search depth
exploration_constant: 1.4    # Exploration constant
```

### Result Processing

#### Selecting Best SQL

```bash
# Select best SQL from MCTS results
bash alphasql_rstar/bash/sconly.sh

# Or specify result directory
bash alphasql_rstar/bash/sconly.sh results/IeSlMRM_dev/Qwen/Qwen2.5-Coder-7B-Instruct
```

#### Evaluating Execution Accuracy

```bash
# Evaluate SQL execution accuracy
bash alphasql_rstar/bash/eval.sh

# Or specify paths
export GT="data/logiccat/IeSlMRM_dev.json"
export PRED="results/IeSlMRM_dev/Qwen/Qwen2.5-Coder-7B-Instruct/selected_sc_only.json"
bash alphasql_rstar/bash/eval.sh
```

---

## Configuration

### MCTS Configuration File

Main configuration file: `alphasql_rstar/config/mcts_logiccat.yaml`

```yaml
tasks_file_path: "data/preprocessed/IeSlMRM_dev/dev/tasks.pkl"
db_root_dir: "data/preprocessed/IeSlMRM_dev"
n_processes: 32
max_rollout_steps: 8
max_depth: 8
exploration_constant: 1.4
save_root_dir: "results/IeSlMRM_dev/Qwen/Qwen2.5-Coder-7B-Instruct"

mcts_model_kwargs:
  model: "Qwen/Qwen2.5-Coder-7B-Instruct"  # Can be overridden via command-line arguments
  temperature: 0.7
  top_p: 0.8
  max_tokens: 2048
```

### Preprocessing Configuration Files

#### LogicCat Preprocessing Configuration

`alphasql_rstar/config/logiccat_preprocess.yaml`

```yaml
data_file_path: "data/logiccat/IeSlMRM_dev.json"
save_root_dir: "data/preprocessed/IeSlMRM_dev"
lsh_threshold: 0.5
embedding_similarity_threshold: 0.6
n_parallel_processes: 8
```

#### Archer Preprocessing Configuration

`alphasql_rstar/config/archer_preprocess.yaml`

```yaml
data_file_path: "data/archer/train.json"
database_root_dir: "data/archer/database"
save_root_dir: "data/preprocessed/archer"
```

---

## Project Structure

```
IESR/
├── alphasql_rstar/          # Core code
│   ├── algorithm/           # MCTS algorithm implementation
│   │   └── mcts/            # MCTS related modules
│   ├── bash/                # Execution scripts
│   │   ├── run_mcts.sh      # MCTS inference script
│   │   ├── run_local_model.sh # Start local vLLM server script
│   │   ├── run_preprocess.sh # Preprocessing script
│   │   ├── eval.sh          # Evaluation script
│   │   └── sconly.sh        # SQL selection script
│   ├── config/              # Configuration files
│   │   ├── mcts_logiccat.yaml
│   │   ├── logiccat_preprocess.yaml
│   │   └── archer_preprocess.yaml
│   ├── database/            # Database related modules
│   ├── llm_call/            # LLM calling modules
│   ├── runner/              # Runner modules
│   └── scripts/             # Utility scripts
├── data/                    # Data directory
│   ├── logiccat/            # LogicCat dataset
│   └── archer/              # Archer dataset
├── logs/                    # Log directory (vLLM server logs)
├── results/                 # Results directory
└── README.md               # This file
```

---

## Frequently Asked Questions

### Q1: How to specify different models?

**A:** There are two ways:

1. **Via command-line arguments** (recommended):
   ```bash
   bash alphasql_rstar/bash/run_mcts.sh "Qwen/Qwen2.5-Coder-7B-Instruct"
   ```

2. **Modify configuration file**:
   Edit the `mcts_model_kwargs.model` field in `alphasql_rstar/config/mcts_logiccat.yaml`

### Q2: How to configure database connections?

**A:** Set via environment variables:

```bash
export MYSQL_HOST="your_host"
export MYSQL_PORT="3306"
export MYSQL_USER="root"
export MYSQL_PASSWORD="your_password"
```

### Q3: How to skip SQL execution validation?

**A:** Set environment variable:

```bash
export ALPHASQL_SKIP_SQL_EXEC=1
```

### Q4: How to handle path issues?

**A:** All scripts have been changed to use relative paths. Ensure you run scripts from the project root directory (`~/IESR`):

```bash
cd ~/IESR
bash alphasql_rstar/bash/run_mcts.sh
```

### Q5: How to adjust parallel process count?

**A:** Modify the `n_processes` parameter in the configuration file:

```yaml
# alphasql_rstar/config/mcts_logiccat.yaml
n_processes: 16  # Adjust according to your CPU core count
```

### Q6: How to start local model server?

**A:** Use the `run_local_model.sh` script:

```bash
# Specify model path
bash alphasql_rstar/bash/run_local_model.sh /path/to/your/model

# Or use environment variables
export MODEL_PATH="/path/to/your/model"
bash alphasql_rstar/bash/run_local_model.sh
```

### Q7: How to adjust vLLM server GPU usage?

**A:** Set via `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Only use GPU 0 and 1
export CUDA_VISIBLE_DEVICES="0,1"
export TP_SIZE=2  # Tensor parallel size must match GPU count
bash alphasql_rstar/bash/run_local_model.sh /path/to/model
```

---

## Contributing

We welcome all forms of contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 Python code style guidelines
- Add necessary comments and docstrings
- Ensure code passes basic syntax checks

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

We thank all developers and researchers who have contributed to this project.

---

## Contact

For questions or suggestions, please contact us via:

- Submit an Issue
- Send a Pull Request

---

<div align="center">

**⭐ If this project helps you, please give us a Star! ⭐**

</div>
