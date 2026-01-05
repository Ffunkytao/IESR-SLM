#!/usr/bin/env python3
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from tqdm import tqdm  # type: ignore

from alphasql_rstar.database.utils import build_table_ddl_statement
from alphasql_rstar.runner.task import Task

# LLM optional imports (fallback to heuristics if unavailable)
try:
    from alphasql_rstar.llm_call.cost_recoder import CostRecorder  # type: ignore
    from alphasql_rstar.llm_call.openai_llm import call_openai  # type: ignore
    from alphasql_rstar.llm_call.local_llm import call_local_llm  # type: ignore
    from alphasql_rstar.llm_call.local_llm import load_local_model  # type: ignore
    from alphasql_rstar.llm_call.prompt_factory import get_prompt  # type: ignore
    from alphasql_rstar.llm_call import local_llm as _llm_module  # type: ignore
except Exception:
    CostRecorder = None
    call_openai = None
    call_local_llm = None
    load_local_model = None  # type: ignore
    get_prompt = None
    _llm_module = None  # type: ignore

from alphasql_rstar.llm_call.embedding_utils import get_embedding_model

from alphasql_mysql.database.mysql_config import MySQLConfig
from alphasql_mysql.database.mysql_connection import MySQLClient
from alphasql_mysql.database.database_manager_mysql import DatabaseManagerMySQL
from alphasql_mysql.database.mysql_lsh_index import MySQLLSHIndex
from alphasql_mysql.database.mysql_sql_parse import extract_db_values_from_sql as extract_db_values_from_sql_mysql


EMBEDDING_MODEL_CALLABLE = get_embedding_model()


@dataclass
class PreprocessConfig:
    data_file_path: str
    save_root_dir: str
    split: str = "dev"
    lsh_threshold: float = 0.5
    lsh_signature_size: int = 128
    lsh_n_gram: int = 3
    lsh_top_k: int = 20
    edit_similarity_threshold: float = 0.3
    embedding_similarity_threshold: float = 0.6
    n_parallel_processes: int = 8
    max_dataset_samples: int = -1
    # LLM options
    use_local_model: bool = True
    local_model_path: Optional[str] = None
    local_model_device: str = "cuda:0"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.2
    # MySQL
    mysql: Optional[Dict[str, Any]] = None

    @classmethod
    def load(cls, p: Path) -> "PreprocessConfig":
        if p.suffix == ".yaml":
            if yaml is None:
                raise RuntimeError("pyyaml not installed")
            d = yaml.safe_load(p.read_text())
        elif p.suffix == ".json":
            d = json.loads(p.read_text())
        else:
            raise ValueError("Unsupported config format, use .yaml or .json")
        return cls(**d)


class PreprocessorMySQL:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self.data: List[Dict[str, Any]] = json.loads(Path(cfg.data_file_path).read_text(encoding="utf-8"))
        if cfg.max_dataset_samples != -1:
            self.data = self.data[: cfg.max_dataset_samples]
        self.save_dir = Path(cfg.save_root_dir) / cfg.split
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # assume each item has db_id
        self.all_db_ids = sorted(list({x["db_id"] for x in self.data}))
        self.tasks = []
        for idx, item in enumerate(self.data):
            qid = (
                item.get("question_id")
                or item.get("qid")
                or item.get("id")
                or item.get("idx")
                or idx
            )
            question = item.get("question") or item.get("question_text") or ""
            # evidence = item.get("evidence") or item.get("hint") or item.get("step") or ""
            evidence = item.get("evidence") or item.get("hint")  or ""
            # support multiple SQL key variants (logiccat uses 'query')
            sql_str = item.get("SQL") or item.get("query") or item.get("sql")
            # Convert difficulty to string if it exists (may be int or str)
            difficulty_raw = item.get("difficulty") or item.get("type")
            difficulty = str(difficulty_raw) if difficulty_raw is not None else None
            self.tasks.append(
                Task(
                    question_id=qid,
                    db_id=item["db_id"],
                    question=question,
                    evidence=evidence,
                    sql=sql_str,
                    difficulty=difficulty,
                )
            )
        if not self.cfg.mysql:
            raise ValueError("MySQL config not found in config file under key 'mysql'")
        self.mysql_config = MySQLConfig.from_dict(self.cfg.mysql)
        self.mysql_client = MySQLClient(self.mysql_config)
        # cost recorder optional
        self.cost_recorder = CostRecorder(self.cfg.model_name) if CostRecorder is not None else None
        # preload local LLM once to avoid repeated loading per task
        if self.cfg.use_local_model and self.cfg.local_model_path and load_local_model is not None:
            try:
                load_local_model(self.cfg.local_model_path, self.cfg.local_model_device)
            except Exception:
                pass
        # mark whether preloaded objects are available
        self._llm_module = _llm_module
        self._use_preloaded_llm = False
        try:
            if self.cfg.use_local_model and self._llm_module is not None:
                if getattr(self._llm_module, "_global_model", None) is not None and getattr(self._llm_module, "_global_tokenizer", None) is not None:
                    self._use_preloaded_llm = True
        except Exception:
            self._use_preloaded_llm = False

    def _generate_with_preloaded_llm(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0) -> str:
        model = getattr(self._llm_module, "_global_model", None)
        tokenizer = getattr(self._llm_module, "_global_tokenizer", None)
        if model is None or tokenizer is None:
            raise RuntimeError("Preloaded LLM not available")
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None) is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"User: {prompt}\nAssistant: "
        import torch  # local import to avoid hard dep if unused
        inputs = tokenizer(text, return_tensors="pt")
        if hasattr(model, 'device'):
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0 or top_p < 1.0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        if top_p < 1.0:
            generation_kwargs["top_p"] = top_p
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated_text

    # -------------------- keyword extraction --------------------
    def get_keywords_for_task(self, task: Task) -> List[str]:
        # Prefer LLM if available
        if get_prompt is not None and (call_local_llm is not None or call_openai is not None):
            prompt = get_prompt("keywords_extraction", {"QUESTION": task.question, "HINT": task.evidence})
            try:
                if self.cfg.use_local_model and self.cfg.local_model_path and (call_local_llm is not None or self._use_preloaded_llm):
                    if self._use_preloaded_llm:
                        resp = self._generate_with_preloaded_llm(
                            prompt,
                            max_tokens=512,
                            temperature=self.cfg.temperature,
                            top_p=1.0,
                        )
                    else:
                        resp = call_local_llm(
                            prompt=prompt,
                            model_path=self.cfg.local_model_path,
                            temperature=self.cfg.temperature,
                            max_tokens=512,
                            device=self.cfg.local_model_device,
                            cost_recorder=self.cost_recorder,
                        )[0]
                elif call_openai is not None:
                    resp = call_openai(
                        prompt,
                        self.cfg.model_name,
                        self.cfg.temperature,
                        cost_recorder=self.cost_recorder,
                    )[0]
                else:
                    raise RuntimeError("No LLM call function available")
                # Parse python list inside markdown fence
                import re

                m = re.search(r"```python\s*\[(.*?)\]\s*```", resp, re.DOTALL)
                if m:
                    s = f"[{m.group(1)}]"
                    raw = eval(s)
                    kws: List[str] = []
                    for kw in raw:
                        kw = str(kw).strip()
                        kws.append(kw)
                        kws.append(kw.replace("/", "-").strip("\"'"))
                        parts = kw.replace("=", " ").replace("(", " ").replace(")", " ").replace("_", " ").split()
                        kws.extend(parts)
                    return sorted(list({x.strip() for x in kws if x.strip()}))
            except Exception:
                pass
        # Heuristic fallback
        import re

        tokens = re.findall(r"[A-Za-z0-9_\-]+", (task.question or "") + " " + (task.evidence or ""))
        return sorted(list({t for t in tokens if len(t) >= 3}))

    # -------------------- gold values (from SQL) --------------------
    def get_gold_relevant_values_for_task(self, task: Task) -> Dict[Tuple[str, str], List[str]]:
        if not task.sql:
            return {}
        db_schema = DatabaseManagerMySQL.get_database_schema(db_id=task.db_id, cache_root_dir=str(self.save_dir), config=self.mysql_config)
        return extract_db_values_from_sql_mysql(task.sql, database_schema=db_schema)

    # -------------------- candidate filtering --------------------
    def filter_by_edit_similarity(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from difflib import SequenceMatcher

        out = []
        for c in candidates:
            sim = SequenceMatcher(None, str(c["value"]), str(c["query"])) .ratio()
            if sim >= self.cfg.edit_similarity_threshold:
                cc = dict(c)
                cc["edit_similarity"] = sim
                out.append(cc)
        return out

    def filter_by_embedding_similarity(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cosine = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        uniq = list({c["value"] for c in candidates} | {c["query"] for c in candidates})
        embs = EMBEDDING_MODEL_CALLABLE.embed_documents(uniq)
        emb_map = {uniq[i]: embs[i] for i in range(len(uniq))}
        out = []
        for c in candidates:
            ve = emb_map[c["value"]]
            qe = emb_map[c["query"]]
            es = cosine(ve, qe)
            if es >= self.cfg.embedding_similarity_threshold:
                cc = dict(c)
                cc["embedding_similarity"] = es
                out.append(cc)
        return out

    def get_relevant_values_for_task(self, task: Task) -> Dict[Tuple[str, str], List[str]]:
        db_schema = DatabaseManagerMySQL.get_database_schema(db_id=task.db_id, cache_root_dir=str(self.save_dir), config=self.mysql_config)
        keywords = self.get_keywords_for_task(task)
        lsh_candidates: List[Dict[str, Any]] = []
        for kw in keywords:
            res = MySQLLSHIndex.query_lsh_index(db_schema, kw, top_k=self.cfg.lsh_top_k, signature_size=self.cfg.lsh_signature_size, n_gram=self.cfg.lsh_n_gram)
            lsh_candidates.extend(res)
        edit_filtered = self.filter_by_edit_similarity(lsh_candidates)
        emb_filtered = self.filter_by_embedding_similarity(edit_filtered)

        final_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        # optional relative filtering
        COEFF = 0.0
        by_col: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for it in emb_filtered:
            key = (it["table_name"], it["column_name"])
            by_col[key].append(it)
        for key, vals in by_col.items():
            max_edit = max(v["edit_similarity"] for v in vals)
            vals = [v for v in vals if v["edit_similarity"] >= COEFF * max_edit]
            max_emb = max(v["embedding_similarity"] for v in vals)
            vals = [v for v in vals if v["embedding_similarity"] >= COEFF * max_emb]
            # keep highest score per distinct value
            best: Dict[str, float] = {}
            for v in vals:
                best[v["value"]] = max(best.get(v["value"], 0.0), v["embedding_similarity"])
            ordered = sorted(best.keys(), key=lambda x: best[x], reverse=True)
            final_map[key] = ordered
        return final_map

    # -------------------- build tasks with schema_context --------------------
    def preprocess_schema_context_for_all_tasks(self, results: List[Dict[Tuple[str, str], List[str]]]) -> List[Task]:
        tasks_with_ctx: List[Task] = []
        for task, rel_map in tqdm(zip(self.tasks, results), total=len(self.tasks), desc="MySQL: build schema context"):
            tcopy = Task(
                question_id=task.question_id,
                db_id=task.db_id,
                question=task.question,
                evidence=task.evidence,
                sql=task.sql,
                difficulty=task.difficulty,
            )
            db_schema = DatabaseManagerMySQL.get_database_schema(task.db_id, str(self.save_dir), self.mysql_config)
            # deepcopy not necessary if we don't mutate shared state deeply
            table_schema_dict = json.loads(json.dumps(db_schema.to_dict()))["tables"]
            # rehydrate to TableSchema objects
            from alphasql_rstar.database.schema import TableSchema as _TS

            table_schema_dict = {k: _TS.from_table_schema_dict(v) for k, v in table_schema_dict.items()}
            for (table_name, column_name), values in rel_map.items():
                if table_name in table_schema_dict and column_name in table_schema_dict[table_name].columns:
                    table_schema_dict[table_name].columns[column_name].value_examples = values[:3]
            schema_context = "\n".join(
                [
                    build_table_ddl_statement(
                        table_schema_dict[tn].to_dict(),
                        add_value_description=True,
                        add_column_description=True,
                        add_value_examples=True,
                        add_expanded_column_name=True,
                    )
                    for tn in table_schema_dict
                ]
            )
            tcopy.schema_context = schema_context
            tcopy.table_schema_dict = table_schema_dict
            tasks_with_ctx.append(tcopy)
        out_pkl = self.save_dir / "tasks.pkl"
        with out_pkl.open("wb") as f:
            pickle.dump(tasks_with_ctx, f)
        return tasks_with_ctx

    # -------------------- LSH building --------------------
    def ensure_lsh_for_all_databases(self) -> None:
        for db_id in tqdm(self.all_db_ids, desc="MySQL: ensure LSH"):
            db_schema = DatabaseManagerMySQL.get_database_schema(db_id=db_id, cache_root_dir=str(self.save_dir), config=self.mysql_config)
            lsh_dir = Path(db_schema.db_directory) / "lsh_index"
            if not lsh_dir.exists():
                MySQLLSHIndex.create_lsh_index(
                    client=self.mysql_client,
                    database_schema=db_schema,
                    threshold=self.cfg.lsh_threshold,
                    signature_size=self.cfg.lsh_signature_size,
                    n_gram=self.cfg.lsh_n_gram,
                )

    # -------------------- pipeline --------------------
    def run(self) -> None:
        # build LSH
        self.ensure_lsh_for_all_databases()
        # get relevant values (serial for simplicity)
        results: List[Dict[Tuple[str, str], List[str]]] = []
        for task in tqdm(self.tasks, total=len(self.tasks), desc="MySQL: relevant values"):
            results.append(self.get_relevant_values_for_task(task))
        with (self.save_dir / "relevant_values_for_all_tasks.pkl").open("wb") as f:
            pickle.dump(results, f)
        # build tasks
        self.preprocess_schema_context_for_all_tasks(results)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="path to yaml/json config for MySQL preprocessing")
    args = ap.parse_args()

    cfg = PreprocessConfig.load(Path(args.config))
    proc = PreprocessorMySQL(cfg)
    proc.run()


if __name__ == "__main__":
    main()
