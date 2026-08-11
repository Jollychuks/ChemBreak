from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch


REQUIRED_MATRIX_COLUMNS = {
    "MATRIX_ID", "HC_ID", "HC_CATEGORY", "HD_ID", "HAZARD_DOMAIN",
    "FIT", "OT_ID", "OUTPUT_TYPE", "ALLOWED_SCENARIOS",
    "DEFAULT_N_CANDIDATES",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_path(project_dir: str | Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(project_dir) / p


def load_matrix(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_MATRIX_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Matrix missing required columns: {sorted(missing)}"
        )
    return df


def split_scenarios(value: Any) -> List[str]:
    if value is None or (
        isinstance(value, float) and pd.isna(value)
    ):
        return []

    text = str(value).strip()
    if not text:
        return []

    return [
        x.strip()
        for x in re.split(r"[|,;]+", text)
        if x.strip()
    ]


def parse_pipe_list(value: Any) -> List[str]:
    if value is None or (
        isinstance(value, float) and pd.isna(value)
    ):
        return []

    text = str(value).strip()
    if not text:
        return []

    return [
        x.strip()
        for x in text.split("|")
        if x.strip()
    ]


def select_rows(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()

    fit = str(config.get("fit", "ALL")).upper()
    if fit != "ALL":
        out = out[
            out["FIT"].astype(str).str.upper() == fit
        ]

    matrix_ids = config.get("matrix_ids") or []
    if matrix_ids:
        wanted = {
            str(x).strip().upper()
            for x in matrix_ids
        }
        out = out[
            out["MATRIX_ID"]
            .astype(str)
            .str.upper()
            .isin(wanted)
        ]

    out = out.reset_index(drop=True)

    start_row = max(
        int(config.get("start_row", 1)) - 1,
        0,
    )

    end_value = config.get("end_row")
    end_row = (
        int(end_value)
        if end_value not in (None, "", 0)
        else len(out)
    )

    return out.iloc[start_row:end_row].reset_index(drop=True)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def stable_int_seed(
    *parts: Any,
    modulus: int = 2_000_000_000,
) -> int:
    raw = "||".join(str(x) for x in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def parse_json_object(text: str) -> Dict[str, Any]:
    clean = str(text).strip()

    clean = re.sub(
        r"^```(?:json)?\s*",
        "",
        clean,
        flags=re.I,
    )
    clean = re.sub(r"\s*```$", "", clean)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise

        data = json.loads(clean[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError(
            "Model output must be a top-level JSON object."
        )

    return data


def append_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    exists = p.exists() and p.stat().st_size > 0

    with p.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not exists:
            writer.writeheader()

        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(
    path: str | Path,
    payload: Dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def existing_ids(
    path: str | Path,
    column: str,
) -> set[str]:
    p = Path(path)

    if not p.exists() or p.stat().st_size == 0:
        return set()

    try:
        df = pd.read_csv(p, usecols=[column])
        return set(df[column].astype(str))
    except Exception:
        return set()


def bullet_text(items: Sequence[str]) -> str:
    if not items:
        return "NONE"

    return "\n".join(
        f"- {x}"
        for x in items
    )


def scenario_details(
    taxonomy: Dict[str, Any],
    scenario_ids: Sequence[str],
) -> str:
    if not scenario_ids:
        return "NONE"

    return " | ".join(
        (
            f"{sc}: "
            f"{taxonomy['SC'].get(sc, 'Unknown scenario')}"
        )
        for sc in scenario_ids
    )


def gpu_report() -> Dict[str, Any]:
    report = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(
                props.total_memory / (1024**3),
                2,
            ),
            "bf16_supported": bool(
                torch.cuda.is_bf16_supported()
            ),
        })

    return report


def git_commit_hash(repo_dir: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_dir),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except Exception:
        return ""


def ensure_same_experiment(
    csv_path: str | Path,
    experiment_id: str,
    allow_mixed: bool,
) -> None:
    p = Path(csv_path)

    if (
        not p.exists()
        or p.stat().st_size == 0
        or allow_mixed
    ):
        return

    try:
        df = pd.read_csv(
            p,
            usecols=["experiment_id"],
        )
    except Exception:
        return

    existing = set(
        df["experiment_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    if existing and existing != {str(experiment_id)}:
        raise RuntimeError(
            f"{p.name} contains experiment IDs {sorted(existing)}, "
            f"but the current run is {experiment_id}. "
            "Use a fresh output directory or enable "
            "allow_mixed_experiments."
        )


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}

    for name in [
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "pandas",
        "huggingface_hub",
        "mistral_common",
    ]:
        try:
            module = __import__(name)
            versions[name] = str(
                getattr(module, "__version__", "unknown")
            )
        except Exception:
            versions[name] = "not-installed"

    return versions
