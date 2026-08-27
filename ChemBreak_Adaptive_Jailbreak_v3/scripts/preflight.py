from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chembreak.config import load_config
from chembreak.runtime_env import (
    configure_cache_environment,
    cleanup_legacy_cache_if_requested,
    disk_snapshot,
    resolved_hf_cache_dir,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/runtime.yaml")
    ap.add_argument("--allow-model-errors", action="store_true")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    runtime = cfg.get("runtime", {})

    cache = configure_cache_environment(runtime)
    cleanup_result = cleanup_legacy_cache_if_requested(runtime)
    cache = configure_cache_environment(runtime)

    # Heavy imports come only after cache routing is fixed.
    import torch
    from huggingface_hub import HfApi
    from chembreak.taskbank import load_taskbank, select_tasks
    from chembreak.vertex import build_client
    from chembreak.prompts import safety_judge_prompt, chemistry_judge_prompt, adjudicator_prompt
    from chembreak.schemas import SAFETY_JUDGE_SCHEMA, CHEMISTRY_JUDGE_SCHEMA, ADJUDICATION_SCHEMA

    print("=== ChemBreak Adaptive Jailbreak v3 preflight ===")
    print("Version:", cfg["version"])
    print("Run mode:", cfg["run_mode"])
    print("Run ID:", cfg.get("run_id", "default"))
    print("Attacker:", cfg["models"]["attacker"]["model"])
    print("Safety judge:", cfg["models"]["safety_judge"]["model"])
    print("Chemistry judge:", cfg["models"]["chemistry_judge"]["model"])
    print("Adjudicator:", cfg["models"]["adjudicator"]["model"])
    print("Attacker receives judge feedback:", bool(cfg.get("adaptive", {}).get("attacker_uses_judge_feedback", False)))
    print("CUDA available:", torch.cuda.is_available())

    cache_disk = disk_snapshot(cache)
    system_disk = disk_snapshot(Path("/"))
    content_disk = disk_snapshot(Path("/content") if Path("/content").exists() else cache)
    print(f"HF cache: {cache_disk['path']}")
    print(f"HF cache disk: {cache_disk['used_gb']} GB used / {cache_disk['total_gb']} GB total | {cache_disk['free_gb']} GB free")
    print(f"System disk: {system_disk['used_gb']} GB used / {system_disk['total_gb']} GB total | {system_disk['free_gb']} GB free")
    if Path("/content").exists():
        print(f"/content disk: {content_disk['used_gb']} GB used / {content_disk['total_gb']} GB total | {content_disk['free_gb']} GB free")
    if cleanup_result.get("removed"):
        print(f"STORAGE CLEANUP | removed legacy system HF cache: {cleanup_result['legacy_path']}")

    min_cache = float(runtime.get("minimum_hf_cache_free_gb", 80))
    min_system = float(runtime.get("minimum_system_free_gb", 5))
    storage_failures = []
    if float(cache_disk["free_gb"]) < min_cache:
        storage_failures.append(f"HF cache disk has {cache_disk['free_gb']} GB free; minimum is {min_cache} GB")
    if float(system_disk["free_gb"]) < min_system:
        storage_failures.append(f"system disk has {system_disk['free_gb']} GB free; minimum is {min_system} GB")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": cfg["version"],
        "run_mode": cfg["run_mode"],
        "run_id": cfg.get("run_id", "default"),
        "git_commit": git_commit(),
        "config_path": str(cfg_path),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "task_bank_uri": cfg["input"]["task_bank_uri"],
        "gcs_output_uri": cfg["gcp"].get("gcs_output_uri", ""),
        "storage": {"hf_cache": cache_disk, "system": system_disk, "content": content_disk, "cleanup": cleanup_result},
        "gpu": {}, "task_bank": {}, "hf_models": [], "model_roles": [], "chemistry_domain_verifier": {},
    }

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        print("GPU:", gpu_name)
        print("GPU memory GB:", gpu_mem)
        manifest["gpu"] = {"available": True, "name": gpu_name, "memory_gb": gpu_mem}
    else:
        print("WARNING: target model execution requires a CUDA GPU.")
        manifest["gpu"] = {"available": False}

    base = Path(runtime.get("local_output_dir", "outputs")) / cfg["version"] / cfg["run_mode"] / str(cfg.get("run_id", "default"))
    work = base / "preflight_work"
    df, mapping = load_taskbank(cfg["input"]["task_bank_uri"], work)
    tasks = select_tasks(df, mapping, cfg)
    print("Task bank rows:", len(df), "selected for mode:", len(tasks))
    print("Detected task-bank mapping:", mapping)
    manifest["task_bank"] = {"rows": len(df), "selected": len(tasks), "mapping": mapping}

    api = HfApi()
    for name, spec in cfg["targets"].items():
        if not spec.get("enabled"):
            continue
        ids = [spec.get("repo_id"), spec.get("base_model_id"), spec.get("adapter_id")]
        for model_id in [x for x in ids if x]:
            entry = {"target": name, "model_id": model_id, "ok": False, "revision": ""}
            try:
                info = api.model_info(model_id)
                entry["ok"] = True
                entry["revision"] = info.sha or "unknown"
                print(f"HF OK | {name} | {model_id} | revision={(info.sha[:8] if info.sha else 'unknown')}")
            except Exception as e:
                entry["error"] = repr(e)
                print(f"HF ERROR | {name} | {model_id} | {e}")
            manifest["hf_models"].append(entry)

    role_failures = []
    harmless_task = {
        "task_id": "PREFLIGHT-HARMLESS",
        "benchmark_prompt": "State whether ordinary water is described consistently without adding new details.",
        "hc_category": "PREFLIGHT",
        "hazard_domain": "PREFLIGHT",
        "output_type": "classification",
    }
    harmless_history = [{"user": "Is water commonly represented as H2O?", "assistant": "Yes."}]
    safety_dummy = {"refusal": 0, "safety_violation": 0, "goal_fulfillment": 0, "actionability": 0, "specificity": 0, "notes": "harmless"}
    chemistry_dummy = {"task_relevance": 5, "chemical_validity": 5, "chemical_plausibility": 5, "internal_consistency": 5, "representation_accuracy": 5, "output_type_fulfillment": 5, "notes": "harmless"}
    verifier_dummy = {"status": "VERIFIED", "checks": []}
    thresholds_dummy = cfg["judging"]["success_thresholds"]

    ok_schema = {
        "type": "object", "additionalProperties": False, "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }

    def role_test(role, client):
        if role == "attacker":
            return client.json('This is a harmless structured-output connectivity test. Return only the requested boolean result.', schema=ok_schema), ["ok"]
        if role == "safety_judge":
            prompt = safety_judge_prompt(harmless_task, harmless_history, "Yes.")
            return client.json(prompt, schema=SAFETY_JUDGE_SCHEMA), ["refusal", "safety_violation", "goal_fulfillment", "actionability", "specificity"]
        if role == "chemistry_judge":
            prompt = chemistry_judge_prompt(harmless_task, "Water is represented as H2O.")
            return client.json(prompt, schema=CHEMISTRY_JUDGE_SCHEMA), ["task_relevance", "chemical_validity", "chemical_plausibility", "internal_consistency", "representation_accuracy", "output_type_fulfillment"]
        if role == "adjudicator":
            prompt = adjudicator_prompt(harmless_task, safety_dummy, chemistry_dummy, verifier_dummy, "Water is represented as H2O.", thresholds_dummy)
            return client.json(prompt, schema=ADJUDICATION_SCHEMA), ["alignment_breach", "effective_chemical_breach", "confidence", "reason"]
        raise ValueError(role)

    for role in ["attacker", "safety_judge", "chemistry_judge", "adjudicator"]:
        spec = cfg["models"][role]
        entry = {
            "role": role, "model": spec.get("model"), "provider": spec.get("provider"),
            "location": spec.get("location", cfg["gcp"].get("location", "global")), "ok": False,
            "structured_output_test": True,
        }
        try:
            client = build_client(spec, cfg["gcp"])
            obj, required = role_test(role, client)
            if not isinstance(obj, dict):
                raise ValueError(f"structured result must be object, got {type(obj).__name__}")
            missing = [k for k in required if k not in obj]
            if missing:
                raise ValueError(f"structured result missing keys: {missing}")
            entry["ok"] = True
            entry["returned_keys"] = sorted(obj.keys())
            if hasattr(client, "diagnostics"):
                entry["diagnostics"] = client.diagnostics()
            print(f"MODEL OK | {role} | {spec['model']} | location={entry['location']} | structured-output keys={entry['returned_keys']}")
        except Exception as e:
            entry["error"] = repr(e)
            role_failures.append(role)
            print(f"MODEL ERROR | {role} | {spec.get('model')} | location={entry['location']} | structured-output test failed: {e}")
        manifest["model_roles"].append(entry)

    verifier = cfg.get("chemistry_domain_verifier", {})
    verifier_entry = {"enabled": bool(verifier.get("enabled", True)), "engine": verifier.get("engine", "deterministic_rdkit_metadata")}
    try:
        import rdkit
        verifier_entry["rdkit_available"] = True
        verifier_entry["rdkit_version"] = getattr(rdkit, "__version__", "unknown")
        print("VERIFIER OK | RDKit", verifier_entry["rdkit_version"])
    except Exception as e:
        verifier_entry["rdkit_available"] = False
        verifier_entry["error"] = repr(e)
        print("VERIFIER WARNING | RDKit unavailable. Metadata-only checks will still run.")
    manifest["chemistry_domain_verifier"] = verifier_entry

    manifest_path = base / runtime.get("public_output_subdir", "public") / "preflight_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Preflight manifest:", manifest_path)
    print("No harmful benchmark task was sent to any target model during preflight.")

    if storage_failures:
        raise SystemExit("Preflight storage check failed: " + "; ".join(storage_failures))
    if role_failures and not args.allow_model_errors:
        raise SystemExit(f"Preflight failed. Unavailable required model roles: {sorted(set(role_failures))}.")
    print("Preflight complete.")


if __name__ == "__main__":
    main()
