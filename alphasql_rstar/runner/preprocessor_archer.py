#!/usr/bin/env python3
"""
Wrapper script for preprocessing archer dataset with YAML config support.
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from alphasql_rstar.runner.preprocessor import Preprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess archer dataset with YAML config")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Extract parameters
    data_file_path = cfg.get("data_file_path")
    database_root_dir = cfg.get("database_root_dir")
    save_root_dir = cfg.get("save_root_dir")
    split = cfg.get("split", "dev")
    lsh_threshold = cfg.get("lsh_threshold", 0.5)
    lsh_signature_size = cfg.get("lsh_signature_size", 128)
    lsh_n_gram = cfg.get("lsh_n_gram", 3)
    lsh_top_k = cfg.get("lsh_top_k", 20)
    edit_similarity_threshold = cfg.get("edit_similarity_threshold", 0.3)
    embedding_similarity_threshold = cfg.get("embedding_similarity_threshold", 0.6)
    n_parallel_processes = cfg.get("n_parallel_processes", 8)
    max_dataset_samples = cfg.get("max_dataset_samples", -1)

    # Create preprocessor
    preprocessor = Preprocessor(
        data_file_path=data_file_path,
        database_root_dir=database_root_dir,
        lsh_threshold=lsh_threshold,
        lsh_signature_size=lsh_signature_size,
        lsh_n_gram=lsh_n_gram,
        lsh_top_k=lsh_top_k,
        edit_similarity_threshold=edit_similarity_threshold,
        embedding_similarity_threshold=embedding_similarity_threshold,
        data_split=split,
        save_root_dir=save_root_dir,
        n_parallel_processes=n_parallel_processes,
        max_dataset_samples=max_dataset_samples,
    )

    # Run preprocessing
    preprocessor.preprocess_lsh_index()
    predicted_relevant_values_for_all_tasks = preprocessor.get_relevant_values_for_all_tasks()
    tasks_with_schema_context = preprocessor.preprocess_schema_context_for_all_tasks()


if __name__ == "__main__":
    main()

