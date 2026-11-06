from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .constants import MANIFEST_SHA256, SOURCE_PROMPTS_SHA256, TASK_COUNT
from .utils import sha256_file

REQUIRED = {"Behavior", "FunctionalCategory", "SemanticCategory"}


def _behavior_hash(text: str) -> str:
    return hashlib.sha256(str(text).strip().encode()).hexdigest()


def load_prompts(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    data = pd.read_csv(source)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Prompt dataset missing columns: {sorted(missing)}")
    if len(data) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} prompt rows, got {len(data)}")
    if data["Behavior"].astype(str).str.strip().str.casefold().duplicated().any():
        raise ValueError("Behavior values must be unique after normalization")
    if sha256_file(source) != SOURCE_PROMPTS_SHA256:
        raise ValueError("Source prompt CSV does not match the locked dataset")
    data["Behavior"] = data["Behavior"].astype(str).str.strip()
    return data


def load_manifest(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    data = pd.read_csv(source)
    if len(data) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} manifest rows, got {len(data)}")
    if data["selection_order"].astype(int).tolist() != list(range(1, TASK_COUNT + 1)):
        raise ValueError("Manifest selection_order must be exactly 1 through 28")
    if data["assignment_id"].astype(str).duplicated().any():
        raise ValueError("Manifest assignment IDs must be unique")
    if sha256_file(source) != MANIFEST_SHA256:
        raise ValueError("Prompt manifest does not match the locked dataset")
    return data


def selected_tasks(prompts_path: str | Path, manifest_path: str | Path) -> pd.DataFrame:
    prompts = load_prompts(prompts_path).reset_index(drop=True)
    manifest = load_manifest(manifest_path).sort_values("selection_order").reset_index(drop=True)
    rows = []
    for index, (manifest_row, source_row) in enumerate(
        zip(manifest.to_dict("records"), prompts.to_dict("records")), start=1
    ):
        if int(manifest_row["source_row"]) != index:
            raise RuntimeError(f"Manifest and source row mismatch at position {index}")
        behavior = str(source_row["Behavior"]).strip()
        if _behavior_hash(behavior) != str(manifest_row["behavior_sha256"]):
            raise RuntimeError(f"Behavior hash mismatch for {manifest_row['assignment_id']}")
        rows.append(
            {
                "selection_order": index,
                "assignment_id": str(manifest_row["assignment_id"]),
                "source_row": index,
                "behavior_sha256": str(manifest_row["behavior_sha256"]),
                "benchmark_prompt": behavior,
                "original_prompt": behavior,
                "goal_intent_anchor": behavior,
                "requested_outcome": "source_behavior",
                "functional_category": str(source_row["FunctionalCategory"]),
                "semantic_category": str(source_row["SemanticCategory"]),
            }
        )
    return pd.DataFrame(rows)
