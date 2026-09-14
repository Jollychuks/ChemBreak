from pathlib import Path
import pandas as pd

from chembreak12.constants import ALL_SPLITS, PRIMARY_SPLIT_SIZES
from chembreak12.dataset import load_task_bank
from chembreak12.integrity import verify_manifest
from chembreak12.partition import build_manifest
from chembreak12.guards import RunAccess, validate_run_access

ROOT = Path(__file__).resolve().parents[1]


def test_partition_is_deterministic_and_disjoint():
    frame = load_task_bank(ROOT / "data" / "final_task_bank.csv")
    a = build_manifest(frame)
    b = build_manifest(frame.sample(frac=1.0, random_state=99).reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b)
    audit = verify_manifest(frame, a)
    assert audit["ok"]
    assert audit["pairwise_overlap_count"] == 0
    assert audit["complete_coverage"]


def test_exact_counts_and_reserve_preserved():
    frame = load_task_bank(ROOT / "data" / "final_task_bank.csv")
    manifest = build_manifest(frame)
    counts = manifest["split"].value_counts().to_dict()
    assert counts == {**PRIMARY_SPLIT_SIZES, "Reserve": 59}
    reserve_ids = set(frame.loc[frame["is_reserve"], "assignment_id"])
    manifest_reserve = set(manifest.loc[manifest["split"] == "Reserve", "assignment_id"])
    assert reserve_ids == manifest_reserve


def test_run_guard():
    validate_run_access(RunAccess("train", "Train"))
    validate_run_access(RunAccess("eval", "Test1"))
    try:
        validate_run_access(RunAccess("train", "Test1"))
    except PermissionError:
        pass
    else:
        raise AssertionError("Training on Test1 should be blocked")
