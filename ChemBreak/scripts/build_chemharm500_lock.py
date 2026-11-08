from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "chemharm500_tasks.xlsx"
MANIFEST = ROOT / "data" / "chemharm500_manifest.csv"
LOCK = ROOT / "data" / "chemharm500_lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(str(value).strip().encode()).hexdigest()


def main() -> None:
    data = pd.read_excel(SOURCE, engine="openpyxl", dtype=str).fillna("")
    required = ["benchmark_prompt", "main_goal"]
    if list(data.columns) != required:
        raise ValueError(f"Expected exactly {required}, got {list(data.columns)}")
    if len(data) != 500:
        raise ValueError(f"Expected 500 tasks, got {len(data)}")
    for column in required:
        data[column] = data[column].astype(str).str.strip()
        if (data[column] == "").any():
            raise ValueError(f"Blank values found in {column}")
    if data["benchmark_prompt"].str.casefold().duplicated().any():
        raise ValueError("Benchmark prompts must be unique")

    ids = [f"CHV15-{index:04d}" for index in range(1, len(data) + 1)]
    manifest = pd.DataFrame({
        "selection_order": range(1, len(data) + 1),
        "assignment_id": ids,
        "source_row": range(1, len(data) + 1),
        "behavior_sha256": [text_hash(value) for value in data["benchmark_prompt"]],
        "goal_sha256": [text_hash(value) for value in data["main_goal"]],
    })
    manifest.to_csv(MANIFEST, index=False, lineterminator="\n")
    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    lock = {
        "namespace": "CHEMBREAK",
        "protocol": "CHEMHARM500_LOCKED_V1",
        "dataset_profile": "chemharm500",
        "source_format": "xlsx",
        "source_columns": required,
        "task_count": len(data),
        "source_prompts_sha256": sha256_file(SOURCE),
        "manifest_sha256": sha256_file(MANIFEST),
        "assignment_ids": ids,
        "assignment_ids_sha256": ids_hash,
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tasks": len(data), "manifest": str(MANIFEST), "lock": str(LOCK)}))


if __name__ == "__main__":
    main()
