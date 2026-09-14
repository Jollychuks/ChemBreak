from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .dataset import load_task_bank
from .integrity import verify_lock, verify_manifest
from .schema import TaskRecord

PHASE_TO_SPLIT = {
    "train": "Train",
    "test1": "Test1",
    "test2": "Test2",
    "test3": "Test3",
    "test4": "Test4",
}


def load_and_validate_task_bank(path: str | Path) -> pd.DataFrame:
    return load_task_bank(path)


def load_partition_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["assignment_id"] = frame["assignment_id"].astype(str).str.strip()
    return frame


def select_phase_tasks(
    frame: pd.DataFrame,
    phase: str,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    if phase not in PHASE_TO_SPLIT:
        raise ValueError(f"Unknown CB12 phase: {phase!r}. Expected one of {sorted(PHASE_TO_SPLIT)}")
    if manifest_path is None:
        raise ValueError("CB12 requires run.partition_manifest_path; runtime resampling is forbidden.")
    manifest = load_partition_manifest(manifest_path)
    verify_manifest(frame, manifest)
    split = PHASE_TO_SPLIT[phase]
    ids = manifest.loc[manifest["split"] == split, ["assignment_id", "split"]]
    selected = ids.merge(frame, on="assignment_id", how="left", validate="one_to_one")
    return selected.sort_values("assignment_id").reset_index(drop=True)


def verify_partition_bundle(
    source_path: str | Path,
    manifest_path: str | Path,
    lock_path: str | Path,
) -> dict[str, object]:
    frame = load_and_validate_task_bank(source_path)
    manifest = load_partition_manifest(manifest_path)
    return verify_lock(
        source_path=source_path,
        frame=frame,
        manifest_path=manifest_path,
        manifest=manifest,
        lock_path=lock_path,
    )


def to_task_records(frame: pd.DataFrame) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for row in frame.fillna("").to_dict(orient="records"):
        records.append(TaskRecord(
            assignment_id=str(row["assignment_id"]), matrix_id=str(row["matrix_id"]),
            hc_id=str(row["hc_id"]), hc_category=str(row["hc_category"]),
            hd_id=str(row["hd_id"]), hazard_domain=str(row["hazard_domain"]),
            ot_id=str(row["ot_id"]), output_type=str(row["output_type"]),
            required_entity=str(row["required_entity"]), benchmark_prompt=str(row["benchmark_prompt"]),
            main_goal=str(row["main_goal"]), chemical_entity=str(row.get("chemical_entity", "")),
            is_reserve=bool(row["is_reserve"]), selected_scenarios=str(row.get("selected_scenarios", "")),
            source_row=row,
        ))
    return records


def selection_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "task_count": len(frame),
        "hc_counts": frame["hc_id"].value_counts().sort_index().to_dict(),
        "hd_counts": frame["hd_id"].value_counts().sort_index().to_dict(),
        "ot_counts": frame["ot_id"].value_counts().sort_index().to_dict(),
        "reserve_count": int(frame["is_reserve"].astype(bool).sum()),
    }


def write_selection(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix(".summary.json").write_text(
        json.dumps(selection_summary(frame), indent=2, sort_keys=True), encoding="utf-8"
    )
