from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .store import load_jsonl


def _first_success_query(obj: dict) -> int | None:
    condition = obj.get("condition")
    result = obj.get("result", {})
    if condition == "C0_direct_single":
        return 1 if result.get("evaluation", {}).get("success") else None
    if condition == "C1_repeated_single":
        for x in result.get("attempts", []):
            if x.get("evaluation", {}).get("success"):
                return int(x.get("attempt", 0))
    else:
        for i, x in enumerate(result.get("turns", []), 1):
            if x.get("evaluation", {}).get("success"):
                return i
    return None


def build_metrics(raw_path: Path, public_dir: Path) -> None:
    rows = []
    for obj in load_jsonl(raw_path):
        if obj.get("status") != "complete":
            continue
        result = obj.get("result", {})
        rows.append({
            "task_id": obj.get("task_id"),
            "target": obj.get("target"),
            "condition": obj.get("condition"),
            "success": bool(result.get("success", False)),
            "queries_used": int(result.get("queries_used", 0)),
            "first_success_query": _first_success_query(obj),
            "route_switches": int(result.get("route_switches", 0) or 0),
        })
    public_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(public_dir / "run_level_metrics.csv", index=False)
    if df.empty:
        summary = pd.DataFrame(columns=["target", "condition", "n", "success_rate", "mean_queries_used"])
    else:
        summary = df.groupby(["target", "condition"], dropna=False).agg(
            n=("task_id", "count"),
            success_rate=("success", "mean"),
            mean_queries_used=("queries_used", "mean"),
            mean_first_success_query=("first_success_query", "mean"),
            mean_route_switches=("route_switches", "mean"),
        ).reset_index()
    summary.to_csv(public_dir / "summary_metrics.csv", index=False)
    (public_dir / "summary_metrics.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
