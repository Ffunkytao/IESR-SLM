#!/usr/bin/env python3
from __future__ import annotations
import argparse
import glob
import json
import os
import pickle
from collections import Counter
from typing import Dict, List
from tqdm import tqdm


def select_sql_majority_vote(results_file_path: str) -> Dict[str, str]:
    """
    SC-only selection for one question.
    Load MCTS pickle, collect all final_sql_query strings, and pick the most frequent.
    """
    question_id = os.path.splitext(os.path.basename(results_file_path))[0]
    with open(results_file_path, "rb") as f:
        results: List[List[object]] = pickle.load(f)

    sqls: List[str] = []
    for path in results:
        try:
            # path[-1] should have attribute final_sql_query
            sql = getattr(path[-1], "final_sql_query", None)
            if isinstance(sql, str) and sql.strip():
                sqls.append(sql.strip())
        except Exception:
            continue

    if not sqls:
        return {"question_id": question_id, "sql": "ERROR"}

    cnt = Counter(sqls)
    # pick the most frequent; tie-break by longest string
    most_common = cnt.most_common()
    top_count = most_common[0][1]
    candidates = [s for s, c in most_common if c == top_count]
    selected = max(candidates, key=len)

    return {"question_id": question_id, "sql": selected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    args = ap.parse_args()

    result_paths = sorted(glob.glob(os.path.join(args.results_dir, "*.pkl")))
    out: Dict[str, str] = {}
    for path in tqdm(result_paths, desc="Selecting (SC-only)"):
        item = select_sql_majority_vote(path)
        out[str(item["question_id"])] = item["sql"]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved selection to {args.output_path} ({len(out)} items)")


if __name__ == "__main__":
    main()

