from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import pandas as pd

from .constants import ALL_SPLITS, PRIMARY_SPLIT_SIZES, PROTOCOL_ID, SEED


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_dataset_sha256(frame: pd.DataFrame) -> str:
    cols = [
        "assignment_id", "is_reserve", "matrix_id", "hc_id", "hd_id", "ot_id", "prompt_sha256"
    ]
    payload = frame[cols].copy().sort_values("assignment_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_manifest(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    required = {"assignment_id", "split", "prompt_sha256", "protocol_id", "seed"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if len(manifest) != len(frame):
        raise ValueError(f"Manifest has {len(manifest)} rows; source bank has {len(frame)}")
    if manifest["assignment_id"].duplicated().any():
        raise ValueError("Manifest assignment_id values are not unique")
    if set(manifest["assignment_id"]) != set(frame["assignment_id"]):
        raise ValueError("Manifest/source assignment_id sets differ")
    unknown = sorted(set(manifest["split"]) - set(ALL_SPLITS))
    if unknown:
        raise ValueError(f"Unknown split names: {unknown}")
    counts = manifest["split"].value_counts().to_dict()
    expected = {**PRIMARY_SPLIT_SIZES, "Reserve": int(frame["is_reserve"].sum())}
    if counts != expected:
        raise ValueError(f"Split counts differ. expected={expected}, observed={counts}")
    if set(manifest["protocol_id"].astype(str)) != {PROTOCOL_ID}:
        raise ValueError("Manifest protocol_id mismatch")
    if set(manifest["seed"].astype(int)) != {SEED}:
        raise ValueError("Manifest seed mismatch")
    merged = frame[["assignment_id", "is_reserve", "prompt_sha256"]].merge(
        manifest[["assignment_id", "split", "prompt_sha256"]],
        on="assignment_id", suffixes=("_source", "_manifest"), validate="one_to_one"
    )
    if not (merged["prompt_sha256_source"] == merged["prompt_sha256_manifest"]).all():
        raise ValueError("Prompt content changed after partitioning")
    if not (merged.loc[merged["is_reserve"], "split"] == "Reserve").all():
        raise ValueError("A source Reserve task was moved out of Reserve")
    if (merged.loc[~merged["is_reserve"], "split"] == "Reserve").any():
        raise ValueError("A primary task was moved into Reserve")
    # Pairwise disjointness follows from unique assignment IDs and exactly one split per row;
    # verify explicitly for auditable output.
    ids = {s: set(manifest.loc[manifest["split"] == s, "assignment_id"]) for s in ALL_SPLITS}
    overlaps: dict[str, list[str]] = {}
    for i, a in enumerate(ALL_SPLITS):
        for b in ALL_SPLITS[i + 1:]:
            overlap = sorted(ids[a] & ids[b])
            if overlap:
                overlaps[f"{a}__{b}"] = overlap
    if overlaps:
        raise ValueError(f"Partition overlap detected: {overlaps}")
    union = set().union(*ids.values())
    if union != set(frame["assignment_id"]):
        raise ValueError("Partition union does not equal source task set")
    return {
        "ok": True,
        "row_count": len(frame),
        "counts": expected,
        "pairwise_overlap_count": 0,
        "complete_coverage": True,
        "reserve_preserved": True,
        "prompt_hashes_match": True,
    }


def write_lock(
    *, source_path: str | Path, frame: pd.DataFrame, manifest_path: str | Path,
    manifest: pd.DataFrame, lock_path: str | Path, audit_path: str | Path,
) -> dict[str, Any]:
    audit = verify_manifest(frame, manifest)
    lock = {
        "protocol_id": PROTOCOL_ID,
        "seed": SEED,
        "source_file": Path(source_path).name,
        "source_file_sha256": sha256_file(source_path),
        "canonical_dataset_sha256": canonical_dataset_sha256(frame),
        "manifest_file": Path(manifest_path).name,
        "manifest_sha256": sha256_file(manifest_path),
        "split_counts": audit["counts"],
        "rules": {
            "reserve": "Preserve source is_reserve=True rows exactly.",
            "training_access": "Train only",
            "evaluation_access": "Test1/Test2/Test3/Test4 only; Reserve requires explicit contingency use.",
            "regeneration": "Forbidden after training begins unless protocol version changes.",
        },
    }
    Path(lock_path).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(audit_path).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def verify_lock(
    *, source_path: str | Path, frame: pd.DataFrame, manifest_path: str | Path,
    manifest: pd.DataFrame, lock_path: str | Path,
) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    verify_manifest(frame, manifest)
    checks = {
        "protocol_id": lock.get("protocol_id") == PROTOCOL_ID,
        "seed": int(lock.get("seed", -1)) == SEED,
        "source_file_sha256": lock.get("source_file_sha256") == sha256_file(source_path),
        "canonical_dataset_sha256": lock.get("canonical_dataset_sha256") == canonical_dataset_sha256(frame),
        "manifest_sha256": lock.get("manifest_sha256") == sha256_file(manifest_path),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"CB12 partition lock failed: {failed}")
    return {"ok": True, "checks": checks, "split_counts": lock["split_counts"]}
