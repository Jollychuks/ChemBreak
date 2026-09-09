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
    operational = clone.get("run", {})
    operational.pop("project_root", None)
    operational.pop("task_bank_path", None)
    operational.pop("output_root", None)
    operational.pop("gcs_checkpoint_uri", None)
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
    if config["run"]["namespace"] != "CB6":
        raise ValueError("This package requires the fresh CB6 namespace.")
    if int(config["retries"].get("structured_output_attempts", 0)) < 2:
        raise ValueError("CB6 requires at least two structured-output validation attempts.")
    phase = config["run"]["phase"]
    expected = {"test": 8, "pilot": 40, "production": 500}
    if phase not in expected:
        raise ValueError(f"Unknown phase: {phase}")
    if int(config["experiment"]["task_count"]) != expected[phase]:
        raise ValueError(f"{phase} must select exactly {expected[phase]} tasks.")
    maximum_budget = int(config["experiment"]["target_query_budget"])
    if maximum_budget != 5:
        raise ValueError("CB6 fixes the maximum target-query budget at five.")
    if "operational_completeness_min" not in config["thresholds"]:
        raise ValueError("CB6 requires thresholds.operational_completeness_min.")
    required = {"C0_DIRECT", "C1_REPEATED_SINGLE", "C2_FIXED_MULTI", "C3_ADAPTIVE_MDP"}
    if set(config["experiment"]["conditions"]) != required:
        raise ValueError("CB6 requires all four registered conditions.")
    budgets = config["experiment"].get("condition_query_budgets")
    expected_budgets = {
        "C0_DIRECT": 1,
        "C1_REPEATED_SINGLE": 5,
        "C2_FIXED_MULTI": 5,
        "C3_ADAPTIVE_MDP": 5,
    }
    if budgets != expected_budgets:
        raise ValueError(
            "CB6 requires a one-query direct control and five-query C1, C2, and C3 conditions."
        )
    allowed_modes = {"strict_schema", "client_validated_json"}
    for role, settings in config["roles"].items():
        mode = settings.get("structured_output_mode", "strict_schema")
        if mode not in allowed_modes:
            raise ValueError(f"Role {role} has unsupported structured-output mode {mode!r}.")
    if not config["run"]["dry_run"] and not config["run"]["live_acknowledgement"]:
        raise ValueError("Live execution requires run.live_acknowledgement=true.")
    target_ids = [item["id"] for item in config["targets"]]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Target IDs must be unique.")
    storage = config.get("storage")
    if not isinstance(storage, dict):
        raise ValueError("CB6 requires an explicit storage section.")
    required_storage = {
        "content_root",
        "storage_root",
        "hf_home",
        "hf_hub_cache",
        "hf_modules_cache",
        "xdg_cache_home",
        "torch_home",
        "torchinductor_cache",
        "triton_cache",
        "cuda_cache",
        "pip_cache",
        "python_packages",
        "temp_dir",
        "offload_dir",
        "preflight_dir",
    }
    missing = sorted(required_storage.difference(storage))
    if missing:
        raise ValueError(f"Missing storage settings: {', '.join(missing)}")
    for target in config["targets"]:
        if not target.get("cache_dir") or not target.get("offload_folder"):
            raise ValueError(f"Target {target['id']} requires cache_dir and offload_folder.")


def resolve_project_path(config_path: str | Path, relative: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    return (project_root / relative).resolve()


def project_root(config: dict[str, Any]) -> Path:
    configured = config.get("run", {}).get("project_root")
    if configured:
        return Path(configured).resolve()
    return Path(config["_config_path"]).resolve().parent.parent


def condition_budget(config: dict[str, Any], condition: str) -> int:
    budgets = config["experiment"]["condition_query_budgets"]
    if condition not in budgets:
        raise ValueError(f"No query budget is registered for condition {condition!r}.")
    return int(budgets[condition])
