from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import pandas as pd

from .constants import BALANCE_WEIGHTS, PRIMARY_SPLITS, PRIMARY_SPLIT_SIZES, PROTOCOL_ID, SEED


def _hash_int(text: str) -> int:
    material = f"{PROTOCOL_ID}|SEED={SEED}|{text}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


def _assignment_key(assignment_id: str) -> str:
    return hashlib.sha256(
        f"{PROTOCOL_ID}|SEED={SEED}|TASK={assignment_id}".encode("utf-8")
    ).hexdigest()


def build_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    reserve = frame.loc[frame["is_reserve"]].copy()
    primary = frame.loc[~frame["is_reserve"]].copy().reset_index(drop=True)
    expected_primary = sum(PRIMARY_SPLIT_SIZES.values())
    if len(primary) != expected_primary:
        raise ValueError(
            f"Primary task count is {len(primary)} but configured split capacities sum to {expected_primary}"
        )

    dims = tuple(BALANCE_WEIGHTS)
    value_counts = {
        d: primary[d].astype(str).value_counts().to_dict()
        for d in dims
    }
    matrix_counts = primary["matrix_id"].astype(str).value_counts().to_dict()
    n = len(primary)
    targets = {
        d: {
            value: {
                split: count * PRIMARY_SPLIT_SIZES[split] / n
                for split in PRIMARY_SPLITS
            }
            for value, count in value_counts[d].items()
        }
        for d in dims
    }
    current = {
        d: {
            value: {split: 0 for split in PRIMARY_SPLITS}
            for value in value_counts[d]
        }
        for d in dims
    }
    remaining = dict(PRIMARY_SPLIT_SIZES)

    primary["_partition_hash_int"] = primary["assignment_id"].map(_hash_int)
    primary["_rarity"] = primary.apply(
        lambda r: sum(1.0 / value_counts[d][str(r[d])] for d in dims)
        + 0.25 / matrix_counts[str(r["matrix_id"])],
        axis=1,
    )
    # Rarest combinations first; cryptographic hash makes ties deterministic and independent of CSV row order.
    order = primary.sort_values(
        ["_rarity", "_partition_hash_int"], ascending=[False, True]
    ).index.tolist()

    assignments: dict[int, str] = {}
    for idx in order:
        row = primary.loc[idx]
        candidates: list[tuple[float, int, int, str]] = []
        for split_index, split in enumerate(PRIMARY_SPLITS):
            if remaining[split] <= 0:
                continue
            cost = 0.0
            for d, weight in BALANCE_WEIGHTS.items():
                value = str(row[d])
                expected = targets[d][value][split]
                before = current[d][value][split] - expected
                after = current[d][value][split] + 1 - expected
                # Incremental normalized squared-error cost.
                cost += weight * ((after * after) - (before * before)) / (expected + 1.0)
            # Mild capacity-fill regularizer; exact capacities remain hard constraints.
            cost += 0.03 * (1.0 - remaining[split] / PRIMARY_SPLIT_SIZES[split])
            tie = _hash_int(f"{row['assignment_id']}|{split}")
            candidates.append((cost, tie, split_index, split))
        if not candidates:
            raise RuntimeError("No split has remaining capacity")
        _, _, _, winner = min(candidates)
        assignments[idx] = winner
        remaining[winner] -= 1
        for d in dims:
            value = str(row[d])
            current[d][value][winner] += 1

    if any(remaining.values()):
        raise RuntimeError(f"Partition capacities were not filled exactly: {remaining}")

    primary["split"] = [assignments[i] for i in primary.index]
    reserve["split"] = "Reserve"
    combined = pd.concat([primary, reserve], ignore_index=True)
    combined["partition_key"] = combined["assignment_id"].map(_assignment_key)
    combined["protocol_id"] = PROTOCOL_ID
    combined["seed"] = SEED
    columns = [
        "assignment_id", "split", "protocol_id", "seed", "partition_key", "prompt_sha256",
        "is_reserve", "matrix_id", "hc_id", "hd_id", "ot_id"
    ]
    return combined[columns].sort_values("assignment_id").reset_index(drop=True)


def write_manifest(manifest: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, lineterminator="\n")


def materialize_split(frame: pd.DataFrame, manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    ids = manifest.loc[manifest["split"] == split, ["assignment_id", "split"]]
    out = ids.merge(frame, on="assignment_id", how="left", validate="one_to_one")
    return out.sort_values("assignment_id").reset_index(drop=True)


def balance_report(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    merged = frame.merge(manifest[["assignment_id", "split"]], on="assignment_id", validate="one_to_one")
    primary = merged[merged["split"].isin(PRIMARY_SPLITS)].copy()
    report: dict[str, Any] = {"split_counts": merged["split"].value_counts().to_dict(), "dimensions": {}}
    for d in BALANCE_WEIGHTS:
        table = pd.crosstab(primary[d], primary["split"])
        table = table.reindex(columns=list(PRIMARY_SPLITS), fill_value=0)
        report["dimensions"][d] = {
            str(index): {split: int(table.loc[index, split]) for split in PRIMARY_SPLITS}
            for index in table.index
        }
    return report
