from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .schema import TASK_BANK_REQUIRED_COLUMNS, TaskRecord


def load_and_validate_task_bank(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(TASK_BANK_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Task bank is missing required columns: {missing}")
    if len(frame) != 500:
        raise ValueError(f"Expected the frozen 500-task ChemHarm bank, found {len(frame)} rows.")
    if frame["assignment_id"].duplicated().any():
        raise ValueError("assignment_id values must be unique.")
    if frame["benchmark_prompt"].duplicated().any():
        raise ValueError("benchmark_prompt values must be unique.")
    lengths = frame["benchmark_prompt"].astype(str).str.split().str.len()
    if not lengths.between(22, 45).all():
        bad = frame.loc[~lengths.between(22, 45), "assignment_id"].tolist()
        raise ValueError(f"Prompts outside the frozen 22-45 word contract: {bad[:10]}")
    if set(frame["hc_id"].dropna().unique()) != {f"HC{i}" for i in range(1, 10)}:
        raise ValueError("The task bank does not contain the complete HC1-HC9 axis.")
    if set(frame["hd_id"].dropna().unique()) != {f"HD{i}" for i in range(1, 9)}:
        raise ValueError("The task bank does not contain the complete HD1-HD8 axis.")
    if set(frame["ot_id"].dropna().unique()) != {f"OT{i}" for i in range(1, 16)}:
        raise ValueError("The task bank does not contain the complete OT1-OT15 axis.")
    return frame


def select_tasks(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count >= len(frame):
        return frame.sort_values("assignment_id").reset_index(drop=True)
    rng = random.Random(seed)
    remaining = list(frame.index)
    rng.shuffle(remaining)
    selected: list[int] = []
    seen_hc: dict[str, int] = {}
    seen_hd: dict[str, int] = {}
    seen_ot: dict[str, int] = {}
    seen_cells: dict[str, int] = {}

    def score(index: int) -> tuple[float, float]:
        row = frame.loc[index]
        novelty = (
            10.0 / (1 + seen_hc.get(row.hc_id, 0))
            + 10.0 / (1 + seen_hd.get(row.hd_id, 0))
            + 7.0 / (1 + seen_ot.get(row.ot_id, 0))
            + 4.0 / (1 + seen_cells.get(row.matrix_id, 0))
        )
        reserve_penalty = 0.5 if bool(row.is_reserve) else 0.0
        return novelty - reserve_penalty, rng.random()

    while len(selected) < count:
        winner = max(remaining, key=score)
        remaining.remove(winner)
        selected.append(winner)
        row = frame.loc[winner]
        for store, key in (
            (seen_hc, row.hc_id),
            (seen_hd, row.hd_id),
            (seen_ot, row.ot_id),
            (seen_cells, row.matrix_id),
        ):
            store[key] = store.get(key, 0) + 1
    return frame.loc[selected].sort_values("assignment_id").reset_index(drop=True)


def to_task_records(frame: pd.DataFrame) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for row in frame.fillna("").to_dict(orient="records"):
        records.append(
            TaskRecord(
                assignment_id=str(row["assignment_id"]),
                matrix_id=str(row["matrix_id"]),
                hc_id=str(row["hc_id"]),
                hc_category=str(row["hc_category"]),
                hd_id=str(row["hd_id"]),
                hazard_domain=str(row["hazard_domain"]),
                ot_id=str(row["ot_id"]),
                output_type=str(row["output_type"]),
                required_entity=str(row["required_entity"]),
                benchmark_prompt=str(row["benchmark_prompt"]),
                main_goal=str(row["main_goal"]),
                chemical_entity=str(row["chemical_entity"]),
                is_reserve=bool(row["is_reserve"]),
                selected_scenarios=str(row.get("selected_scenarios", "")),
                source_row=row,
            )
        )
    return records


def selection_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "task_count": int(len(frame)),
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

