from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from .benchmark import (
    load_and_validate_task_bank,
    select_phase_tasks,
    selection_summary,
)
from .config import load_config, project_root, validate_config
from .policy import AdaptiveQPolicy
from .prompts import SYSTEM_CONTROLLED_RESEARCH
from .providers import RoleClients
from .storage import configure_content_storage, verify_content_storage
from .targets import make_target
from .utils import require_live_gate, utc_now

REQUIRED_IMPORTS = (
    "pandas", "yaml", "pydantic", "scipy", "transformers", "accelerate", "peft",
    "einops", "huggingface_hub", "google.genai", "google.cloud.storage", "openai", "rdkit",
)

ROLE_PROMPTS = {
    "observer": "Classify the harmless response 'I cannot help with unsafe requests.'",
    "adaptive_actor": "Select a harmless test action and create a short harmless test utterance.",
    "safety_verifier": "Classify the harmless refusal 'I cannot help with unsafe requests.'",
    "chemistry_verifier": "Score this harmless statement: 'No chemistry procedure is supplied.'",
    "adjudicator": "Resolve a harmless test disagreement with a non-success result.",
}


def _check_imports() -> dict[str, str]:
    values = {}
    for name in REQUIRED_IMPORTS:
        module = importlib.import_module(name)
        values[name] = str(getattr(module, "__version__", "installed"))
    return values


def _check_gpu(require_gpu: bool) -> dict[str, Any]:
    import torch

    available = torch.cuda.is_available()
    if require_gpu and not available:
        raise RuntimeError("A CUDA GPU is required for the local target models.")
    return {
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else [],
        "torch_version": torch.__version__,
    }


def _check_roles(config: dict[str, Any], project_id: str | None) -> list[dict[str, Any]]:
    clients = RoleClients(config, project_id)
    records = []
    for role in config["roles"]:
        clients.call_json(role, ROLE_PROMPTS[role], SYSTEM_CONTROLLED_RESEARCH)
        calls = clients.drain_call_history()
        final = calls[-1]
        if not final.get("schema_applied"):
            raise RuntimeError(f"{role} did not accept its response schema.")
        records.append({
            "role": role, "provider": final.get("provider"), "model": final.get("model"),
            "schema_applied": True, "structured_attempt": final.get("structured_attempt"), "status": "ok",
        })
    return records


def _probe_tokenizers(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for settings in config["targets"]:
        target = make_target(settings, bool(config["run"]["dry_run"]))
        try:
            records.append(target.load_tokenizer_only())
        finally:
            target.unload()
    return records


def run_preflight(
    config_path: str | Path, load_targets: bool = False, probe_tokenizers: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    configure_content_storage(config)
    root = project_root(config)
    storage_report = verify_content_storage(config, root)
    require_live_gate(config)
    project_id = None
    if not config["run"]["dry_run"]:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
        if not project_id:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT before live preflight.")
    bank_setting = Path(config["run"]["task_bank_path"])
    bank_path = bank_setting if bank_setting.is_absolute() else root / bank_setting
    frame = load_and_validate_task_bank(bank_path)
    selected = select_phase_tasks(frame, config["run"]["phase"])
    policy = AdaptiveQPolicy(config["policy"], int(config["run"]["seed"]))
    result = {
        "status": "ok", "checked_at_utc": utc_now(), "platform": platform.platform(),
        "phase": config["run"]["phase"], "dry_run": config["run"]["dry_run"],
        "storage": storage_report, "imports": _check_imports(),
        "gpu": _check_gpu(require_gpu=not config["run"]["dry_run"]),
        "task_bank": selection_summary(frame), "selected_subset": selection_summary(selected),
        "policy": policy.summary(),
        "roles": _check_roles(config, project_id),
    }
    if probe_tokenizers:
        result["tokenizers"] = _probe_tokenizers(config)
    if load_targets:
        loaded = []
        for settings in config["targets"]:
            target = make_target(settings, bool(config["run"]["dry_run"]))
            target.load()
            target.unload()
            loaded.append(settings["id"])
        result["loaded_targets"] = loaded
    output = Path(config["storage"]["preflight_dir"]) / "preflight_report.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
