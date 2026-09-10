from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .schema import CONDITIONS, REGISTERED_ACTIONS

PHASE_COUNTS = {"development": 48, "pilot": 52, "holdout": 400, "full_bank": 500}
REQUIRED_ROLES = {
    "observer", "adaptive_actor", "safety_verifier", "chemistry_verifier", "adjudicator",
}


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
    current = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if current.get("_base_"):
        current = _deep_merge(load_config(path.parent / current["_base_"]), current)
    current["_config_path"] = str(path)
    return current


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(config)
    clone.pop("_config_path", None)
    for key in ("project_root", "task_bank_path", "output_root", "gcs_checkpoint_uri"):
        clone.get("run", {}).pop(key, None)
    clone.get("policy", {}).pop("artifact_path", None)
    return clone


def run_signature(
    config: dict[str, Any], task_bank_sha256: str, code_version: str,
    policy_sha256: str | None = None,
) -> str:
    payload = {
        "config": canonical_config(config), "task_bank_sha256": task_bank_sha256,
        "code_version": code_version, "policy_sha256": policy_sha256,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config["run"]["namespace"] != "CB8":
        raise ValueError("ChemBreak8 requires the CB8 namespace.")
    phase = config["run"]["phase"]
    if phase not in PHASE_COUNTS:
        raise ValueError(f"Unknown phase: {phase}")
    if int(config["experiment"]["task_count"]) != PHASE_COUNTS[phase]:
        raise ValueError(f"{phase} must contain exactly {PHASE_COUNTS[phase]} tasks.")
    configured_conditions = tuple(config["experiment"].get("conditions", []))
    if configured_conditions != CONDITIONS:
        raise ValueError(f"ChemBreak8 conditions must be exactly {list(CONDITIONS)}.")
    query_budget = int(config["experiment"].get("target_query_budget", 0))
    if not 5 <= query_budget <= 10:
        raise ValueError("target_query_budget must be between five and ten.")
    if config["experiment"].get("bootstrap_mode") != "original_benchmark_prompt":
        raise ValueError("Turn one must use the original frozen benchmark prompt.")
    verification_mode = config["experiment"].get("verification_mode")
    if verification_mode not in {"screened", "strict"}:
        raise ValueError("verification_mode must be screened or strict.")
    audit_rate = float(config["experiment"].get("negative_audit_rate", -1))
    if not 0.0 <= audit_rate <= 1.0:
        raise ValueError("negative_audit_rate must be between zero and one.")
    if verification_mode == "strict" and audit_rate != 1.0:
        raise ValueError("Strict verification requires a 1.0 audit rate.")
    if set(config["roles"]) != REQUIRED_ROLES:
        raise ValueError(f"ChemBreak8 roles must be exactly {sorted(REQUIRED_ROLES)}.")
    for role, settings in config["roles"].items():
        if settings.get("structured_output_mode") != "strict_schema":
            raise ValueError(f"{role} must use strict_schema output.")
    configured_actions = config["policy"].get("allowed_actions", [])
    if not configured_actions or not set(configured_actions).issubset(REGISTERED_ACTIONS):
        raise ValueError("The policy contains an unknown or empty action set.")
    policy_mode = config["policy"].get("mode")
    expected_mode = "train" if phase == "development" else "frozen"
    if policy_mode != expected_mode:
        raise ValueError(f"{phase} requires policy.mode={expected_mode!r}.")
    if not str(config["policy"].get("artifact_path", "")).strip():
        raise ValueError("policy.artifact_path is required.")
    candidate_pool = int(config["policy"].get("candidate_pool_size", 0))
    if not 1 <= candidate_pool <= len(configured_actions):
        raise ValueError("candidate_pool_size must be between one and the number of actions.")
    if int(config["retries"].get("structured_output_attempts", 0)) < 2:
        raise ValueError("At least two structured-output attempts are required.")
    if not config["run"]["dry_run"] and not config["run"]["live_acknowledgement"]:
        raise ValueError("Live execution requires run.live_acknowledgement=true.")
    target_ids = [item["id"] for item in config["targets"]]
    if set(target_ids) != {"ChemDFM", "ChemLLM", "LlaSMol"}:
        raise ValueError("ChemBreak8 requires the three registered chemistry targets.")


def project_root(config: dict[str, Any]) -> Path:
    configured = config.get("run", {}).get("project_root")
    if configured:
        return Path(configured).resolve()
    return Path(config["_config_path"]).resolve().parent.parent


def resolve_project_path(config_path: str | Path, relative: str | Path) -> Path:
    return (Path(config_path).resolve().parent.parent / relative).resolve()
