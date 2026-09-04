from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: object, length: int = 16) -> str:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("Parsed JSON was not an object.")
                return value
    raise ValueError("Incomplete JSON object in model output.")


def retry_call(
    fn: Callable[[], T],
    attempts: int,
    initial_backoff: float,
    maximum_backoff: float,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    delay = initial_backoff
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if on_retry:
                on_retry(attempt, exc)
            time.sleep(delay)
            delay = min(maximum_backoff, delay * 2)
    assert last_error is not None
    raise last_error


def require_live_gate(config: dict[str, Any]) -> None:
    if config["run"]["dry_run"]:
        return
    if os.environ.get("CHEMBREAK_ENABLE_LIVE") != "YES":
        raise RuntimeError(
            "Live execution is locked. Set CHEMBREAK_ENABLE_LIVE=YES only inside the controlled research environment."
        )

