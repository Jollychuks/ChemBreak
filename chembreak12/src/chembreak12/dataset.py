from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd

from .constants import REQUIRED_COLUMNS


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("is_reserve contains a null value")
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse is_reserve value: {value!r}")


def normalize_prompt(text: object) -> str:
    return " ".join(str(text).split()).strip()


def prompt_sha256(text: object) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def load_task_bank(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Task bank missing required columns: {missing}")
    if len(frame) != 500:
        raise ValueError(f"ChemBreak 12 expects exactly 500 source rows; found {len(frame)}")
    frame = frame.copy()
    frame["assignment_id"] = frame["assignment_id"].astype(str).str.strip()
    frame["is_reserve"] = frame["is_reserve"].map(_parse_bool)
    if frame["assignment_id"].eq("").any():
        raise ValueError("assignment_id contains an empty value")
    if frame["assignment_id"].duplicated().any():
        dup = frame.loc[frame["assignment_id"].duplicated(), "assignment_id"].tolist()
        raise ValueError(f"Duplicate assignment_id values detected: {dup[:10]}")
    normalized = frame["benchmark_prompt"].map(normalize_prompt)
    if normalized.eq("").any():
        raise ValueError("benchmark_prompt contains an empty value")
    if normalized.duplicated().any():
        dup = frame.loc[normalized.duplicated(), "assignment_id"].tolist()
        raise ValueError(f"Duplicate normalized benchmark prompts detected: {dup[:10]}")
    frame["prompt_sha256"] = frame["benchmark_prompt"].map(prompt_sha256)
    return frame
