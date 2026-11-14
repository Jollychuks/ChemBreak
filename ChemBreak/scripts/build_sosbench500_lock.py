from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "sosbench500_tasks.csv"
MANIFEST = ROOT / "data" / "sosbench500_manifest.csv"
LOCK = ROOT / "data" / "sosbench500_lock.json"

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def text_hash(value: str) -> str:
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()

def main() -> None:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["goal", "original_term", "subject"]:
            raise ValueError(f"Unexpected source columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 500:
        raise ValueError(f"Expected 500 chemistry requests, got {len(rows)}")
    if any(not row["goal"].strip() or not row["original_term"].strip()
           or row["subject"].strip().lower() != "chemistry" for row in rows):
        raise ValueError("Missing text or non-chemistry record in source")
    if len({r["goal"].strip().casefold() for r in rows}) != 500:
        raise ValueError("Dataset contains duplicate objectives")

    ids = [f"SOSCHEM-{i:04d}" for i in range(1, 501)]
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["selection_order", "assignment_id", "source_row",
                         "behavior_sha256", "goal_sha256"])
        for index, (assignment_id, row) in enumerate(zip(ids, rows), 1):
            h = text_hash(row["goal"])
            writer.writerow([index, assignment_id, index, h, h])
    lock = {
        "namespace": "CHEMBREAK",
        "protocol": "SOSBENCH500_LOCKED_V1",
        "dataset_profile": "sosbench500",
        "source_format": "csv",
        "source_columns": ["goal", "original_term", "subject"],
        "source_url": "https://huggingface.co/datasets/SOSBench/SOSBench",
        "task_count": 500,
        "source_prompts_sha256": sha256_file(SOURCE),
        "manifest_sha256": sha256_file(MANIFEST),
        "assignment_ids": ids,
        "assignment_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"profile": "sosbench500", "tasks": len(rows),
                      "source_sha256": sha256_file(SOURCE)}))

if __name__ == "__main__":
    main()
