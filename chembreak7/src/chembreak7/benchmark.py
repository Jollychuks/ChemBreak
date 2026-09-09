from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .schema import TASK_BANK_REQUIRED_COLUMNS, TaskRecord

DEVELOPMENT_IDS = (
    "CBV15C-0113",
    "CBV15C-0270",
    "CBV15C-0289",
    "CBV15C-0373",
    "CBV15C-0416",
    "CBV15C-0417",
    "CBV15C-0420",
    "CBV15C-0426",
)
PILOT_SEED = 7102026


def load_and_validate_task_bank(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(TASK_BANK_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Task bank is missing required columns: {missing}")
    if len(frame) != 500:
        raise ValueError(f"Expected 500 frozen ChemHarm tasks, found {len(frame)}.")
    if frame["assignment_id"].duplicated().any():
        raise ValueError("assignment_id values must be unique.")
    if frame["benchmark_prompt"].duplicated().any():
        raise ValueError("benchmark_prompt values must be unique.")
    return frame


def _balanced_select(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count >= len(frame):
        return frame.sort_values("assignment_id").reset_index(drop=True)
    rng = random.Random(seed)
    remaining = list(frame.index)
    rng.shuffle(remaining)
    selected: list[int] = []
    seen: dict[str, dict[str, int]] = {name: {} for name in ("hc_id", "hd_id", "ot_id", "matrix_id")}

    def score(index: int) -> tuple[float, float]:
        row = frame.loc[index]
        novelty = sum(
            weight / (1 + seen[column].get(str(row[column]), 0))
            for column, weight in (("hc_id", 10.0), ("hd_id", 10.0), ("ot_id", 7.0), ("matrix_id", 4.0))
        )
        return novelty - (0.5 if bool(row.is_reserve) else 0.0), rng.random()

    while len(selected) < count:
        winner = max(remaining, key=score)
        remaining.remove(winner)
        selected.append(winner)
        row = frame.loc[winner]
        for column, counts in seen.items():
            key = str(row[column])
            counts[key] = counts.get(key, 0) + 1
    return frame.loc[selected].sort_values("assignment_id").reset_index(drop=True)


def task_partitions(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    development = frame[frame.assignment_id.astype(str).isin(DEVELOPMENT_IDS)].copy()
    if set(development.assignment_id.astype(str)) != set(DEVELOPMENT_IDS):
        raise RuntimeError("The frozen bank does not contain every registered development task.")
    remainder = frame[~frame.assignment_id.astype(str).isin(DEVELOPMENT_IDS)].copy()
    pilot = _balanced_select(remainder, 40, PILOT_SEED)
    pilot_ids = set(pilot.assignment_id.astype(str))
    holdout = remainder[~remainder.assignment_id.astype(str).isin(pilot_ids)].copy()
    return {
        "development": development.sort_values("assignment_id").reset_index(drop=True),
        "pilot": pilot,
        "holdout": holdout.sort_values("assignment_id").reset_index(drop=True),
        "full_bank": frame.sort_values("assignment_id").reset_index(drop=True),
    }


def select_phase_tasks(frame: pd.DataFrame, phase: str) -> pd.DataFrame:
    partitions = task_partitions(frame)
    if phase not in partitions:
        raise ValueError(f"Unknown task phase: {phase}")
    return partitions[phase].copy()


def to_task_records(frame: pd.DataFrame) -> list[TaskRecord]:
    records = []
    for row in frame.fillna("").to_dict(orient="records"):
        records.append(TaskRecord(
            assignment_id=str(row["assignment_id"]), matrix_id=str(row["matrix_id"]),
            hc_id=str(row["hc_id"]), hc_category=str(row["hc_category"]),
            hd_id=str(row["hd_id"]), hazard_domain=str(row["hazard_domain"]),
            ot_id=str(row["ot_id"]), output_type=str(row["output_type"]),
            required_entity=str(row["required_entity"]), benchmark_prompt=str(row["benchmark_prompt"]),
            main_goal=str(row["main_goal"]), chemical_entity=str(row["chemical_entity"]),
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
