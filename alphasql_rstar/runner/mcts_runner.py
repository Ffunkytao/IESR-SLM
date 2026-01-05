from alphasql_rstar.algorithm.mcts.mcts import MCTSSolver
from alphasql_rstar.algorithm.mcts.reward import MajorityVoteRewardModel
from alphasql_rstar.runner.task import Task
from alphasql_rstar.config.mcts_config import MCTSConfig
from pathlib import Path
from typing import Union
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import yaml
from alphasql_rstar.llm_call.openai_llm import DEFAULT_COST_RECORDER
import json
import random
from dotenv import load_dotenv  
import os
import traceback

load_dotenv(override=True)

try:
    import weave
    if "WANDB_API_KEY" in os.environ:
        weave.init("mcts_runner")
    else:
        print("WANDB_API_KEY is not set in environment variables, will not use it")
except ImportError:
    print("Weave is not installed, will not use it")

class MCTSRunner:
    def __init__(self, config: Union[MCTSConfig, str]):
        if isinstance(config, str):
            config_path = Path(config)
            assert config_path.exists(), f"Config file {config_path} does not exist"
            if config_path.suffix == ".json":
                self.config = MCTSConfig.model_validate_json(config_path.read_text())
            elif config_path.suffix == ".yaml":
                self.config = MCTSConfig.model_validate(yaml.safe_load(config_path.read_text()))
            else:
                raise ValueError(f"Unsupported config file extension: {config_path.suffix}")
        else:
            self.config = config
            
        if not Path(self.config.save_root_dir).exists():
            Path(self.config.save_root_dir).mkdir(parents=True, exist_ok=True)

        random.seed(self.config.random_seed)
        
    def run_one_task(self, task: Task) -> str:
        print(f"[进度] 开始处理任务 ID: {task.question_id}")
        mcts_solver = MCTSSolver(
            db_root_dir=self.config.db_root_dir,
            task=task,
            max_rollout_steps=self.config.max_rollout_steps,
            max_depth=self.config.max_depth,
            exploration_constant=self.config.exploration_constant,
            save_root_dir=self.config.save_root_dir,
            llm_kwargs=self.config.mcts_model_kwargs,
            reward_model=MajorityVoteRewardModel(self.config.reward_model_kwargs)
        )
        try:
            mcts_solver.solve()
            print(f"[进度] 完成处理任务 ID: {task.question_id}")
        except Exception as e:
            print("-" * 100)
            print(f"Error solving task {task.question_id}: {e}")
            traceback.print_exc()
            print(f"The task {task.question_id} has been given up")
            print("-" * 100)
        DEFAULT_COST_RECORDER.print_profile()
    
    def run_all_tasks(self):
        with open(self.config.tasks_file_path, "rb") as f:
            tasks = pickle.load(f)
            
        if self.config.subset_file_path:
            print(f"Using subset file {self.config.subset_file_path} to filter tasks")
            with open(self.config.subset_file_path, "r") as f:
                subset_data = json.load(f)
                subset_ids = [item["question_id"] for item in subset_data]
                tasks = [task for task in tasks if task.question_id in subset_ids]
            print(f"Filtered {len(tasks)} tasks from {len(tasks)} tasks")
        
        done_task_ids = []
        for pkl_file in Path(self.config.save_root_dir).glob("*.pkl"):
            done_task_ids.append(int(pkl_file.stem))
        print(f"Ignore done task ids: {done_task_ids}")
        tasks = [task for task in tasks if task.question_id not in done_task_ids]
        
        with open(Path(self.config.save_root_dir) / "config.json", "w") as f:
            print(f"Saving config to {Path(self.config.save_root_dir) / 'config.json'}")
            json.dump(self.config.model_dump(), f, indent=4)

        print(f"There are {len(tasks)} tasks to solve")
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            list(tqdm(executor.map(self.run_one_task, tasks), total=len(tasks), desc="Solving tasks"))

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="MCTS Runner for AlphaSQL")
    parser.add_argument("config_path", type=str, help="Path to the MCTS configuration file")
    parser.add_argument("--model", type=str, default=None, 
                       help="Model name to use (overrides config file). Example: Qwen/Qwen2.5-Coder-7B-Instruct")
    
    args = parser.parse_args()
    
    config_path = args.config_path
    runner = MCTSRunner(config=config_path)
    
    # 如果通过命令行指定了模型，覆盖配置文件中的模型设置
    if args.model:
        print(f"[INFO] Overriding model from command line: {args.model}")
        runner.config.mcts_model_kwargs["model"] = args.model
        # 更新保存目录以反映模型名称
        model_name_short = args.model.split("/")[-1] if "/" in args.model else args.model
        # 从保存目录路径中提取基础路径，然后替换模型名称部分
        import re
        # 匹配路径中的模型名称部分并替换
        save_dir = runner.config.save_root_dir
        # 如果路径中包含模型名称，替换它；否则在路径末尾添加模型名称
        if "Qwen" in save_dir:
            # 替换任何现有的模型名称
            save_dir = re.sub(r"Qwen[^/]*", f"Qwen/{model_name_short}", save_dir)
        else:
            # 如果路径中没有模型名称，在适当位置添加
            if "/Qwen/" in save_dir:
                save_dir = save_dir.replace("/Qwen/", f"/Qwen/{model_name_short}/")
            else:
                save_dir = f"{save_dir}/Qwen/{model_name_short}"
        runner.config.save_root_dir = save_dir
        # 确保保存目录存在
        if not Path(runner.config.save_root_dir).exists():
            Path(runner.config.save_root_dir).mkdir(parents=True, exist_ok=True)
    
    runner.run_all_tasks()
