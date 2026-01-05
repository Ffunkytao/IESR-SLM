#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Tuple
from decimal import Decimal
from tqdm import tqdm  # type: ignore

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from alphasql_mysql.database.mysql_config import MySQLConfig
from alphasql_mysql.database.mysql_connection import MySQLClient
from alphasql_mysql.database.mysql_sql_execution import (
    execute_sql_with_timeout,
    SQLExecutionResultType,
)


def load_mysql_config(p: str) -> MySQLConfig:
    with open(p, "r", encoding="utf-8") as f:
        if p.endswith(".yaml") or p.endswith(".yml"):
            if yaml is None:
                raise RuntimeError("pyyaml not installed to parse yaml config")
            d = yaml.safe_load(f)
        else:
            d = json.load(f)
    if isinstance(d, dict) and "mysql" in d:
        d = d["mysql"]
    return MySQLConfig.from_dict(d)


def rows_to_key(rows: List[Tuple[Any, ...]]) -> Tuple[Tuple[Any, ...], ...]:
    # order-insensitive key (set-like). Note: duplicates are ignored.
    # If you want to keep duplicates, sort rows and return tuple of tuples.
    try:
        return tuple(sorted(set(tuple(r) for r in rows)))
    except TypeError:
        # Non-hashable items: fallback to string repr
        return tuple(sorted(set(tuple(str(x) for x in r) for r in rows)))


def to_jsonable(obj: Any) -> Any:
    """Recursively convert SQL results (which may contain Decimal, tuples, etc.) to JSON-serializable types."""
    if isinstance(obj, Decimal):
        # Prefer float; fallback to str for very large/precise decimals if needed
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser(description="Evaluate execution accuracy against MySQL")
    ap.add_argument("--gt_data", type=str, required=True, help="Path to ground-truth data.json")
    ap.add_argument("--pred_json", type=str, required=True, help="Path to predictions JSON {question_id: sql}")
    ap.add_argument("--mysql_config", type=str, required=True, help="YAML/JSON with mysql credentials or top-level mysql key")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--output_dir", type=str, default="", help="Directory to write eval_details.json. Defaults to the directory of pred_json if empty.")
    args = ap.parse_args()

    # Load data and predictions
    gt_items: List[Dict[str, Any]] = json.loads(open(args.gt_data, "r", encoding="utf-8").read())
    preds: Dict[str, str] = json.loads(open(args.pred_json, "r", encoding="utf-8").read())

    # Build index from data by question_id-like field
    gt_map: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(gt_items):
        qid = (
            item.get("question_id")
            or item.get("qid")
            or item.get("id")
            or item.get("idx")
            or i
        )
        gt_map[str(qid)] = item

    # Prepare MySQL client
    mysql_cfg = load_mysql_config(args.mysql_config)
    client = MySQLClient(mysql_cfg)

    # Evaluate
    total, matched, gt_fail, pred_fail, skipped = 0, 0, 0, 0, 0
    details: List[Dict[str, Any]] = []

    for qid, pred_sql in tqdm(preds.items(), desc="Evaluating"):
        item = gt_map.get(str(qid))
        if not item:
            skipped += 1
            details.append({"question_id": qid, "status": "skipped_missing_gt"})
            continue
        db_id = item["db_id"]
        # ground truth sql key variants
        gt_sql = item.get("SQL") or item.get("query") or item.get("sql")
        if not gt_sql or not isinstance(gt_sql, str):
            skipped += 1
            details.append({"question_id": qid, "status": "skipped_no_gt_sql"})
            continue

        total += 1
        gt_res = execute_sql_with_timeout(client, db_id, gt_sql, timeout=args.timeout)
        if gt_res.result_type != SQLExecutionResultType.SUCCESS:
            gt_fail += 1
            details.append({
                "question_id": qid,
                "status": "gt_error",
                "gt_error": gt_res.error_message,
            })
            continue

        pred_res = execute_sql_with_timeout(client, db_id, pred_sql, timeout=args.timeout)
        if pred_res.result_type != SQLExecutionResultType.SUCCESS:
            pred_fail += 1
            details.append({
                "question_id": qid,
                "status": "pred_error",
                "pred_error": pred_res.error_message,
            })
            continue

        # Compare rows (order-insensitive, duplicate-insensitive)
        gt_key = rows_to_key(gt_res.result or [])
        pred_key = rows_to_key(pred_res.result or [])

        if gt_key == pred_key:
            matched += 1
            details.append({"question_id": qid, "status": "match"})
        else:
            details.append({
                "question_id": qid,
                "status": "mismatch",
                "gt_rows": to_jsonable(gt_res.result),
                "pred_rows": to_jsonable(pred_res.result),
            })

    acc = matched / total if total > 0 else 0.0

    summary = {
        "total_compared": total,
        "matched": matched,
        "accuracy": acc,
        "gt_fail": gt_fail,
        "pred_fail": pred_fail,
        "skipped": skipped,
    }

    print("==== Execution Accuracy Summary ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Save details next to prediction file
    out_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.pred_json))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_details.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": to_jsonable(details)}, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved detailed report to {out_path}")


if __name__ == "__main__":
    main()

