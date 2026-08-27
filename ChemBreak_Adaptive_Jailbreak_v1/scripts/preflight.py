from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chembreak.config import load_config
from chembreak.taskbank import load_taskbank, select_tasks
from chembreak.vertex import build_client


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/runtime.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    print("=== ChemBreak Adaptive Jailbreak preflight ===")
    print("Version:", cfg["version"])
    print("Run mode:", cfg["run_mode"])
    print("Run ID:", cfg.get("run_id", "default"))
    print("CUDA available:", torch.cuda.is_available())

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
        "gpu": {},
        "task_bank": {},
        "hf_models": [],
        "vertex_roles": [],
    }

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        print("GPU:", gpu_name)
        print("GPU memory GB:", gpu_mem)
        manifest["gpu"] = {"available": True, "name": gpu_name, "memory_gb": gpu_mem}
    else:
        print("WARNING: target model execution requires a CUDA GPU. Asset preparation can still use Vertex models.")
        manifest["gpu"] = {"available": False}

    base = Path(cfg["runtime"].get("local_output_dir", "outputs")) / cfg["version"] / cfg["run_mode"] / str(cfg.get("run_id", "default"))
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

    for role in ["attacker", "safety_judge", "chemistry_judge", "adjudicator"]:
        spec = cfg["models"][role]
        entry = {"role": role, "model": spec.get("model"), "provider": spec.get("provider"), "ok": False}
        try:
            client = build_client(spec, cfg["gcp"])
            obj = client.json('Return exactly this JSON object and nothing else: {"ok": true}')
            entry["ok"] = bool(obj.get("ok", False))
            print(f"VERTEX OK | {role} | {spec['model']} | {obj}")
        except Exception as e:
            entry["error"] = repr(e)
            print(f"VERTEX ERROR | {role} | {spec.get('model')} | {e}")
        manifest["vertex_roles"].append(entry)

    manifest_path = base / cfg["runtime"].get("public_output_subdir", "public") / "preflight_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Preflight manifest:", manifest_path)
    print("Preflight complete. No harmful benchmark task was sent to any target model.")


if __name__ == "__main__":
    main()
