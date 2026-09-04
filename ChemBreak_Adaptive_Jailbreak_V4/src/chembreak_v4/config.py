from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base_name = current.get("_base_")
    if base_name:
        base = load_config(path.parent / base_name)
        current = _deep_merge(base, current)
    current["_config_path"] = str(path)
    return current


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(config)
    clone.pop("_config_path", None)
    return clone


def run_signature(config: dict[str, Any], task_bank_sha256: str, code_version: str) -> str:
    payload = {
        "config": canonical_config(config),
        "task_bank_sha256": task_bank_sha256,
        "code_version": code_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config["run"]["namespace"] != "CBJ4":
        raise ValueError("This package requires the fresh CBJ4 namespace.")
    phase = config["run"]["phase"]
    expected = {"test": 8, "pilot": 40, "production": 500}
    if phase not in expected:
        raise ValueError(f"Unknown phase: {phase}")
    if int(config["experiment"]["task_count"]) != expected[phase]:
        raise ValueError(f"{phase} must select exactly {expected[phase]} tasks.")
    budget = int(config["experiment"]["target_query_budget"])
    if budget != 5:
        raise ValueError("CBJ4 fixes every condition to the same five-query budget.")
    required = {"C0_DIRECT", "C1_REPEATED_SINGLE", "C2_FIXED_MULTI", "C3_ADAPTIVE_MDP"}
    if set(config["experiment"]["conditions"]) != required:
        raise ValueError("CBJ4 requires all four registered conditions.")
    if not config["run"]["dry_run"] and not config["run"]["live_acknowledgement"]:
        raise ValueError("Live execution requires run.live_acknowledgement=true.")
    target_ids = [item["id"] for item in config["targets"]]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Target IDs must be unique.")


def resolve_project_path(config_path: str | Path, relative: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    return (project_root / relative).resolve()

