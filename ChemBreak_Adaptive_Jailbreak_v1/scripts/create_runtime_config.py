from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser(description="Create a local runtime config without modifying the committed template.")
    ap.add_argument("--template", default="configs/gcp.yaml")
    ap.add_argument("--output", default="configs/runtime.yaml")
    ap.add_argument("--project", required=True)
    ap.add_argument("--run-mode", choices=["test", "pilot", "production"], required=True)
    ap.add_argument("--run-id", required=True, help="Stable ID used for resume, for example test_001 or production_001")
    ap.add_argument("--task-bank-uri", required=True)
    ap.add_argument("--gcs-output-uri", required=True)
    args = ap.parse_args()

    src = Path(args.template)
    dst = Path(args.output)
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["run_mode"] = args.run_mode
    cfg["run_id"] = args.run_id
    cfg.setdefault("gcp", {})["project"] = args.project
    cfg["gcp"]["gcs_output_uri"] = args.gcs_output_uri.rstrip("/")
    cfg.setdefault("input", {})["task_bank_uri"] = args.task_bank_uri
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Runtime config written: {dst}")
    print(f"  mode: {args.run_mode}")
    print(f"  run id: {args.run_id}")
    print(f"  task bank: {args.task_bank_uri}")
    print(f"  outputs: {args.gcs_output_uri.rstrip('/')}")


if __name__ == "__main__":
    main()
