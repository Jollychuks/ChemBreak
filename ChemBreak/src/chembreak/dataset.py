from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .utils import sha256_file


DATASET_PROFILES = {
    "prompts28": {
        "task_count": 28,
        "prompts_path": "data/prompts.csv",
        "manifest_path": "data/prompts28_manifest.csv",
        "lock_path": "data/prompts28_lock.json",
        "experiment_revision": "CHEMBREAK_PROGRESSIVE_RHS_OFS_PROMPTS28_V4_1",
    },
    "chemharm500": {
        "task_count": 500,
        "prompts_path": "data/chemharm500_tasks.xlsx",
        "manifest_path": "data/chemharm500_manifest.csv",
        "lock_path": "data/chemharm500_lock.json",
        "experiment_revision": "CHEMBREAK_PROGRESSIVE_RHS_OFS_CHEMHARM500_V1",
    },
}


def dataset_profile_spec(profile: str) -> dict:
    key = str(profile).strip().lower()
    if key not in DATASET_PROFILES:
        raise ValueError(f"Unknown dataset profile {profile!r}; choose from {sorted(DATASET_PROFILES)}")
    return dict(DATASET_PROFILES[key])


def _text_hash(text: str) -> str:
    return hashlib.sha256(str(text).strip().encode()).hexdigest()


def _read_source(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(source, engine="openpyxl")
    raise ValueError(f"Unsupported task source format: {source.suffix}")


def load_lock(path: str | Path) -> dict:
    lock = json.loads(Path(path).read_text())
    required = {
        "namespace", "protocol", "dataset_profile", "task_count",
        "source_prompts_sha256", "manifest_sha256", "assignment_ids",
        "assignment_ids_sha256",
    }
    missing = sorted(required - set(lock))
    if missing:
        raise ValueError(f"Dataset lock missing fields: {missing}")
    if str(lock["namespace"]) != "CHEMBREAK":
        raise ValueError("Dataset lock namespace must be CHEMBREAK")
    if int(lock["task_count"]) < 1:
        raise ValueError("Dataset lock task_count must be positive")
    if len(lock["assignment_ids"]) != int(lock["task_count"]):
        raise ValueError("Dataset lock assignment ID count does not match task_count")
    return lock


def load_prompts(path: str | Path, lock: dict) -> pd.DataFrame:
    source = Path(path)
    data = _read_source(source)
    profile = str(lock["dataset_profile"])
    expected_count = int(lock["task_count"])
    if len(data) != expected_count:
        raise ValueError(f"Expected {expected_count} prompt rows, got {len(data)}")
    if sha256_file(source) != str(lock["source_prompts_sha256"]):
        raise ValueError("Source prompt file does not match the selected dataset lock")

    if profile == "prompts28":
        required = {"Behavior", "FunctionalCategory", "SemanticCategory"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Prompt dataset missing columns: {sorted(missing)}")
        normalized = pd.DataFrame({
            "benchmark_prompt": data["Behavior"].astype(str).str.strip(),
            "main_goal": data["Behavior"].astype(str).str.strip(),
            "requested_outcome": "source_behavior",
            "functional_category": data["FunctionalCategory"].astype(str),
            "semantic_category": data["SemanticCategory"].astype(str),
        })
    elif profile == "chemharm500":
        required = {"benchmark_prompt", "main_goal"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"ChemHarm dataset missing columns: {sorted(missing)}")
        normalized = pd.DataFrame({
            "benchmark_prompt": data["benchmark_prompt"].astype(str).str.strip(),
            "main_goal": data["main_goal"].astype(str).str.strip(),
            "requested_outcome": "source_behavior",
            "functional_category": "chemharm",
            "semantic_category": "chemical_biological",
        })
    else:
        raise ValueError(f"Unsupported dataset profile in lock: {profile}")

    if (normalized["benchmark_prompt"] == "").any() or (normalized["main_goal"] == "").any():
        raise ValueError("Task prompts and main goals must not be blank")
    if normalized["benchmark_prompt"].str.casefold().duplicated().any():
        raise ValueError("Benchmark prompts must be unique after normalization")
    return normalized


def load_manifest(path: str | Path, lock: dict) -> pd.DataFrame:
    source = Path(path)
    data = pd.read_csv(source)
    expected_count = int(lock["task_count"])
    required = {"selection_order", "assignment_id", "source_row", "behavior_sha256"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset manifest missing columns: {sorted(missing)}")
    if len(data) != expected_count:
        raise ValueError(f"Expected {expected_count} manifest rows, got {len(data)}")
    if data["selection_order"].astype(int).tolist() != list(range(1, expected_count + 1)):
        raise ValueError(f"Manifest selection_order must be exactly 1 through {expected_count}")
    if data["assignment_id"].astype(str).duplicated().any():
        raise ValueError("Manifest assignment IDs must be unique")
    if sha256_file(source) != str(lock["manifest_sha256"]):
        raise ValueError("Prompt manifest does not match the selected dataset lock")
    if data["assignment_id"].astype(str).tolist() != [str(x) for x in lock["assignment_ids"]]:
        raise ValueError("Manifest assignment IDs do not match the selected dataset lock")
    return data


def selected_tasks(
    prompts_path: str | Path,
    manifest_path: str | Path,
    lock_path: str | Path,
) -> pd.DataFrame:
    lock = load_lock(lock_path)
    prompts = load_prompts(prompts_path, lock).reset_index(drop=True)
    manifest = load_manifest(manifest_path, lock).sort_values("selection_order").reset_index(drop=True)
    rows = []
    for index, (manifest_row, source_row) in enumerate(
        zip(manifest.to_dict("records"), prompts.to_dict("records")), start=1
    ):
        if int(manifest_row["source_row"]) != index:
            raise RuntimeError(f"Manifest and source row mismatch at position {index}")
        behavior = str(source_row["benchmark_prompt"]).strip()
        main_goal = str(source_row["main_goal"]).strip()
        if _text_hash(behavior) != str(manifest_row["behavior_sha256"]):
            raise RuntimeError(f"Behavior hash mismatch for {manifest_row['assignment_id']}")
        if "goal_sha256" in manifest and _text_hash(main_goal) != str(manifest_row["goal_sha256"]):
            raise RuntimeError(f"Main-goal hash mismatch for {manifest_row['assignment_id']}")
        rows.append({
            "selection_order": index,
            "assignment_id": str(manifest_row["assignment_id"]),
            "source_row": index,
            "dataset_profile": str(lock["dataset_profile"]),
            "behavior_sha256": str(manifest_row["behavior_sha256"]),
            "benchmark_prompt": behavior,
            "original_prompt": behavior,
            "goal_intent_anchor": main_goal,
            "requested_outcome": str(source_row["requested_outcome"]),
            "functional_category": str(source_row["functional_category"]),
            "semantic_category": str(source_row["semantic_category"]),
        })
    return pd.DataFrame(rows)
