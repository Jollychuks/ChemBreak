from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

TRAIN_SPLITS = {"Train"}
EVAL_SPLITS = {"Test1", "Test2", "Test3", "Test4"}


@dataclass(frozen=True)
class RunAccess:
    mode: str
    split: str
    reserve_reason: str | None = None


def validate_run_access(access: RunAccess) -> None:
    mode = access.mode.lower().strip()
    split = access.split.strip()
    if mode == "train":
        if split not in TRAIN_SPLITS:
            raise PermissionError(
                f"CB12 training is locked to Train only; requested split={split!r}"
            )
        return
    if mode == "eval":
        if split not in EVAL_SPLITS:
            raise PermissionError(
                f"CB12 ordinary evaluation is restricted to Test1-Test4; requested split={split!r}"
            )
        return
    if mode == "reserve":
        if split != "Reserve":
            raise PermissionError("Reserve mode can access only the Reserve partition")
        if not (access.reserve_reason or "").strip():
            raise PermissionError("Reserve access requires a recorded contingency reason")
        return
    raise ValueError("mode must be one of: train, eval, reserve")


def write_run_access_record(access: RunAccess, path: str | Path) -> None:
    validate_run_access(access)
    payload = {"mode": access.mode, "split": access.split, "reserve_reason": access.reserve_reason}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
