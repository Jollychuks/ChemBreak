from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import os
import random
import time

from .attack_assets import prepare_task_assets
from .controlled import direct_single, repeated_single, fixed_multi
from .adaptive import run_adaptive
from .config import load_config
from .gcs import is_gs, sync_down_prefix, sync_up_dir
from .metrics import build_metrics
from .progress import Progress
from .store import append_jsonl, load_jsonl, completed_keys
from .taskbank import load_taskbank, select_tasks
from .targets import build_target
from .vertex import build_client


def _dirs(cfg):
    base = Path(cfg["runtime"].get("local_output_dir", "outputs")) / cfg["version"] / cfg["run_mode"] / str(cfg.get("run_id", "default"))
    return base, base / cfg["runtime"].get("restricted_output_subdir", "restricted"), base / cfg["runtime"].get("public_output_subdir", "public")


def _clients(cfg):
    gcp = cfg["gcp"]
    m = cfg["models"]
    return (
        build_client(m["attacker"], gcp),
        build_client(m["safety_judge"], gcp),
        build_client(m["chemistry_judge"], gcp),
        build_client(m["adjudicator"], gcp),
    )


def _load_selected(cfg, base):
    df, mapping = load_taskbank(cfg["input"]["task_bank_uri"], base)
    tasks = select_tasks(df, mapping, cfg)
    return tasks, mapping


def _load_assets(path: Path) -> dict[str, dict[str, Any]]:
    return {str(x["task_id"]): x["assets"] for x in load_jsonl(path) if x.get("status") == "complete"}


def prepare(cfg):
    base, restricted, public = _dirs(cfg)
    base.mkdir(parents=True, exist_ok=True)
    if cfg["gcp"].get("gcs_output_uri"):
        sync_down_prefix(cfg["gcp"]["gcs_output_uri"], base)
    tasks, mapping = _load_selected(cfg, base)
    attacker, _, _, _ = _clients(cfg)
    path = restricted / "attack_assets.jsonl"
    done = completed_keys(path, ("task_id",))
    p = Progress("PREPARE", len(tasks))
    for task in tasks:
        key = (task["task_id"],)
        if key in done:
            p.step(f"skip {task['task_id']}")
            continue
        try:
            assets = prepare_task_assets(task, attacker, cfg)
            append_jsonl(path, {"status": "complete", "run_id": cfg.get("run_id", "default"), "task_id": task["task_id"], "assets": assets})
            done.add(key)
            note = f"saved {task['task_id']}"
        except Exception as e:
            append_jsonl(restricted / "errors.jsonl", {"stage": "prepare", "task_id": task["task_id"], "error": repr(e), "time": time.time()})
            note = f"ERROR {task['task_id']}: {e}"
        p.step(note)
        if cfg["gcp"].get("gcs_output_uri") and p.done % int(cfg["gcp"].get("checkpoint_every_tasks", 1)) == 0:
            sync_up_dir(base, cfg["gcp"]["gcs_output_uri"])

    selected_ids = [str(t["task_id"]) for t in tasks]
    completed_after = _load_assets(path)
    missing_after = [task_id for task_id in selected_ids if task_id not in completed_after]
    public.mkdir(parents=True, exist_ok=True)
    (public / "selected_task_ids.json").write_text(json.dumps(selected_ids, indent=2), encoding="utf-8")
    (public / "prepare_summary.json").write_text(json.dumps({
        "selected_tasks": len(selected_ids),
        "complete_assets": len(selected_ids) - len(missing_after),
        "missing_assets": len(missing_after),
        "missing_task_ids": missing_after,
        "max_generation_attempts_per_asset": int(cfg.get("asset_preparation", {}).get("max_generation_attempts", 3)),
    }, indent=2), encoding="utf-8")
    if cfg["gcp"].get("gcs_output_uri"):
        sync_up_dir(base, cfg["gcp"]["gcs_output_uri"])
    if missing_after:
        raise RuntimeError(
            f"PREPARE incomplete: {len(missing_after)} selected tasks still lack complete attack assets after retries. "
            f"Rerun prepare with the same RUN_MODE/RUN_ID. Example: {missing_after[:3]}"
        )
    print(f"[PREPARE] complete: {len(selected_ids)}/{len(selected_ids)} selected tasks have frozen attack assets.", flush=True)


def execute(cfg, only_target: str | None = None, only_section: str = "all"):
    base, restricted, public = _dirs(cfg)
    if cfg["gcp"].get("gcs_output_uri"):
        sync_down_prefix(cfg["gcp"]["gcs_output_uri"], base)
    tasks, _ = _load_selected(cfg, base)
    assets = _load_assets(restricted / "attack_assets.jsonl")
    missing = [t["task_id"] for t in tasks if t["task_id"] not in assets]
    if missing:
        raise RuntimeError(f"Missing attack assets for {len(missing)} tasks. Run prepare first. Example: {missing[:3]}")
    attacker, safety, chemistry, adjudicator = _clients(cfg)
    judges = (safety, chemistry, adjudicator)
    raw_path = restricted / "executions.jsonl"
    done = completed_keys(raw_path, ("task_id", "target", "condition"))
    target_specs = [(n, s) for n, s in cfg["targets"].items() if s.get("enabled") and (only_target in (None, "all", n))]
    conditions = []
    if only_section in ("all", "controlled"):
        if cfg["controlled"].get("run_direct_single"): conditions.append("C0_direct_single")
        if cfg["controlled"].get("run_repeated_single"): conditions.append("C1_repeated_single")
        if cfg["controlled"].get("run_fixed_multi"): conditions.append("C2_fixed_multi")
    if only_section in ("all", "adaptive") and cfg["adaptive"].get("run"): conditions.append("C3_adaptive_chembreak")
    total = len(tasks) * len(target_specs) * len(conditions)
    p = Progress("EXECUTE", total)
    for target_name, spec in target_specs:
        display = spec.get("display_name", target_name)
        print(f"[MODEL] Loading {display} once for this target block...", flush=True)
        load_start = time.time()
        target = None
        try:
            target = build_target(target_name, spec, cfg["target_generation"])
            elapsed = time.time() - load_start
            print(f"[MODEL] {display} loaded in {elapsed:.1f}s. It will be reused across all tasks/conditions for this target.", flush=True)
        except Exception as e:
            append_jsonl(restricted / "errors.jsonl", {"stage": "target_load", "target": target_name, "error": repr(e), "time": time.time()})
            remaining_units = len(tasks) * len(conditions)
            for _ in range(remaining_units):
                p.step(f"ERROR target load {target_name}: {e}")
            continue
        try:
            for task in tasks:
                asset = assets[task["task_id"]]
                for condition in conditions:
                    key = (task["task_id"], target_name, condition)
                    if key in done:
                        p.step(f"skip {target_name} {task['task_id']} {condition}")
                        continue
                    try:
                        if condition == "C0_direct_single":
                            result = direct_single(task, target, judges, cfg, target_name=display)
                        elif condition == "C1_repeated_single":
                            result = repeated_single(task, asset, target, judges, cfg, target_name=display)
                        elif condition == "C2_fixed_multi":
                            result = fixed_multi(task, asset, target, judges, cfg, target_name=display)
                        elif condition == "C3_adaptive_chembreak":
                            result = run_adaptive(task, asset, target, attacker, judges, cfg, target_name=display)
                        else:
                            raise ValueError(condition)
                        append_jsonl(raw_path, {"status": "complete", "run_id": cfg.get("run_id", "default"), "task_id": task["task_id"], "target": target_name, "condition": condition, "result": result})
                        note = f"{target_name} {task['task_id']} {condition} success={result.get('success')} q={result.get('queries_used')}"
                    except Exception as e:
                        append_jsonl(restricted / "errors.jsonl", {"stage": "execute", "task_id": task["task_id"], "target": target_name, "condition": condition, "error": repr(e), "time": time.time()})
                        note = f"ERROR {target_name} {task['task_id']} {condition}: {e}"
                    p.step(note)
                    if cfg["gcp"].get("gcs_output_uri") and p.done % int(cfg["gcp"].get("checkpoint_every_tasks", 1)) == 0:
                        sync_up_dir(base, cfg["gcp"]["gcs_output_uri"])
        finally:
            if target is not None:
                print(f"[MODEL] Releasing {display} before loading the next target...", flush=True)
                target.close()
    build_metrics(raw_path, public)
    if cfg["gcp"].get("gcs_output_uri"):
        sync_up_dir(base, cfg["gcp"]["gcs_output_uri"])


def metrics(cfg):
    base, restricted, public = _dirs(cfg)
    build_metrics(restricted / "executions.jsonl", public)
    if cfg["gcp"].get("gcs_output_uri"):
        sync_up_dir(base, cfg["gcp"]["gcs_output_uri"])
