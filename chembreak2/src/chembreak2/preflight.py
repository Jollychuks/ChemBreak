from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from .benchmark import load_and_validate_task_bank, select_tasks, selection_summary
from .config import load_config, resolve_project_path, validate_config
from .prompts import SYSTEM_CONTROLLED_RESEARCH
from .providers import RoleClients
from .storage import configure_content_storage, verify_content_storage
from .targets import make_target
from .utils import extract_json_object, require_live_gate, utc_now


REQUIRED_IMPORTS = (
    "pandas",
    "yaml",
    "pydantic",
    "rich",
    "scipy",
    "transformers",
    "accelerate",
    "peft",
    "google.genai",
    "google.cloud.storage",
    "openai",
    "rdkit",
)


def _check_imports() -> dict[str, str]:
    results: dict[str, str] = {}
    for name in REQUIRED_IMPORTS:
        module = importlib.import_module(name)
        results[name] = str(getattr(module, "__version__", "installed"))
    return results


def _check_gpu(require_gpu: bool) -> dict[str, Any]:
    import torch

    available = torch.cuda.is_available()
    if require_gpu and not available:
        raise RuntimeError("A CUDA GPU is required for the three local target models.")
    return {
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if available
        else [],
        "torch_version": torch.__version__,
    }


def _check_repositories(config: dict[str, Any]) -> list[dict[str, str]]:
    from huggingface_hub import model_info

    records: list[dict[str, str]] = []
    for target in config["targets"]:
        for kind, model_id in (("model", target["model"]), ("adapter", target.get("adapter"))):
            if not model_id:
                continue
            info = model_info(model_id)
            records.append(
                {
                    "target_id": target["id"],
                    "kind": kind,
                    "model_id": model_id,
                    "revision": str(info.sha or "unknown"),
                }
            )
    return records


def _check_roles(config: dict[str, Any], project_id: str | None) -> list[dict[str, Any]]:
    clients = RoleClients(config, project_id)
    records = []
    for role in config["roles"]:
        call_role = role
        if role == "planner":
            expected_key = "action_id"
        elif role == "realizer":
            expected_key = "utterance"
        elif role == "safety_judge":
            expected_key = "success"
        elif role == "chemistry_judge":
            expected_key = "chemical_relevance"
        else:
            expected_key = "success"
        response = clients.call(
            role,
            f'Return a JSON object containing the key "{expected_key}". This is a benign connectivity check.',
            SYSTEM_CONTROLLED_RESEARCH,
            call_role=call_role,
        )
        data = extract_json_object(response.text)
        if expected_key not in data:
            raise ValueError(f"{role} preflight response omitted {expected_key}.")
        records.append(
            {"role": role, "provider": response.provider, "model": response.model, "status": "ok"}
        )
    return records


def run_preflight(config_path: str | Path, load_targets: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    configure_content_storage(config)
    project_root = Path(config["_config_path"]).parent.parent.resolve()
    storage_report = verify_content_storage(config, project_root)
    require_live_gate(config)
    project_id = None
    if not config["run"]["dry_run"]:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
        if not project_id:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT before live preflight.")
    bank_path = resolve_project_path(config["_config_path"], config["run"]["task_bank_path"])
    frame = load_and_validate_task_bank(bank_path)
    selected = select_tasks(frame, int(config["experiment"]["task_count"]), int(config["run"]["seed"]))
    result: dict[str, Any] = {
        "status": "ok",
        "checked_at_utc": utc_now(),
        "platform": platform.platform(),
        "phase": config["run"]["phase"],
        "dry_run": config["run"]["dry_run"],
        "storage": storage_report,
        "imports": _check_imports(),
        "gpu": _check_gpu(require_gpu=not config["run"]["dry_run"]),
        "task_bank": selection_summary(frame),
        "selected_subset": selection_summary(selected),
        "roles": _check_roles(config, project_id),
    }
    if not config["run"]["dry_run"]:
        result["model_repositories"] = _check_repositories(config)
    if load_targets:
        loaded = []
        for target_settings in config["targets"]:
            target = make_target(target_settings, bool(config["run"]["dry_run"]))
            target.load()
            target.unload()
            loaded.append(target_settings["id"])
        result["loaded_targets"] = loaded
    output = Path(config["storage"]["preflight_dir"]) / "preflight_report.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
