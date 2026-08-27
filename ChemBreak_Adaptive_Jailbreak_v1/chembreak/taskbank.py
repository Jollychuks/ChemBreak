from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import random
import pandas as pd
from .gcs import is_gs, download_file


ALIASES = {
    "task_id": ["task_id", "behavior_id", "benchmark_id", "assignment_id", "id"],
    "benchmark_prompt": ["benchmark_prompt", "prompt", "task", "task_text", "behavior", "goal"],
    "hc_category": ["hc_category", "hc", "category", "capability_category"],
    "hazard_domain": ["hazard_domain", "hd", "domain"],
    "output_type": ["output_type", "ot", "deliverable_type"],
    "entity": ["entity", "required_entity", "chemical_entity", "substance"],
    "scenario": ["scenario", "scenario_constraint", "context_constraint", "allowed_scenario"],
}


def _find_col(columns: list[str], aliases: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def load_taskbank(uri: str, work_dir: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    path = Path(uri)
    if is_gs(uri):
        path = download_file(uri, work_dir / "input" / "final_task_bank.csv")
    if not path.exists():
        raise FileNotFoundError(f"Task bank not found: {uri}")
    df = pd.read_csv(path)
    mapping = {k: _find_col(list(df.columns), v) for k, v in ALIASES.items()}
    if not mapping["benchmark_prompt"]:
        raise ValueError(f"Could not find benchmark prompt column. Available columns: {list(df.columns)}")
    if not mapping["task_id"]:
        ids = []
        for i, prompt in enumerate(df[mapping["benchmark_prompt"]].fillna("")):
            h = hashlib.sha256(str(prompt).encode()).hexdigest()[:10]
            ids.append(f"CB-AUTO-{i+1:04d}-{h}")
        df["__auto_task_id"] = ids
        mapping["task_id"] = "__auto_task_id"
    return df, mapping


def normalize_task(row: pd.Series, mapping: dict[str, str | None]) -> dict[str, Any]:
    def get(key: str) -> str:
        col = mapping.get(key)
        if not col:
            return ""
        val = row.get(col, "")
        return "" if pd.isna(val) else str(val).strip()
    return {k: get(k) for k in ALIASES}


def select_tasks(df: pd.DataFrame, mapping: dict[str, str | None], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    mode = cfg["run_mode"]
    limits = cfg["sampling"]
    limit = limits["test_limit"] if mode == "test" else limits["pilot_limit"] if mode == "pilot" else limits["production_limit"]
    records = [normalize_task(r, mapping) for _, r in df.iterrows()]
    if limit is None or len(records) <= int(limit):
        return records
    rng = random.Random(int(cfg.get("seed", 0)))
    strata_keys = [k for k in limits.get("stratify_columns", []) if mapping.get(k)]
    if not strata_keys:
        rng.shuffle(records)
        return records[: int(limit)]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for rec in records:
        key = tuple(rec.get(k, "") for k in strata_keys)
        groups.setdefault(key, []).append(rec)
    for vals in groups.values():
        rng.shuffle(vals)
    selected: list[dict[str, Any]] = []
    keys = list(groups)
    rng.shuffle(keys)
    while len(selected) < int(limit) and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < int(limit):
                selected.append(groups[key].pop())
    return selected
