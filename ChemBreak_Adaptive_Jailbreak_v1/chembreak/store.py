from __future__ import annotations
from pathlib import Path
import json
import os
from typing import Any


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def completed_keys(path: Path, fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    keys = set()
    for obj in load_jsonl(path):
        if obj.get("status") == "complete":
            keys.add(tuple(str(obj.get(f, "")) for f in fields))
    return keys
