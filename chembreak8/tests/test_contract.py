from __future__ import annotations

from pathlib import Path

from chembreak8 import CHECKPOINT_PROTOCOL_VERSION, __version__
from chembreak8.benchmark import (
    DEVELOPMENT_IDS,
    load_and_validate_task_bank,
    task_partitions,
)
from chembreak8.config import PHASE_COUNTS, load_config, run_signature, validate_config
from chembreak8.schema import CONDITIONS
from chembreak8.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_bank_and_disjoint_partitions():
    bank = ROOT / "data" / "final_task_bank.csv"
    assert sha256_file(bank) == "62df773ce8c4a252bd23350fcd3a8e83fc670864efb0304fe8c65d21d3c7d6ff"
    frame = load_and_validate_task_bank(bank)
    parts = task_partitions(frame)
    assert {name: len(value) for name, value in parts.items()} == PHASE_COUNTS
    assert set(parts["development"].assignment_id) == set(DEVELOPMENT_IDS)
    assert set(parts["development"].assignment_id).isdisjoint(parts["pilot"].assignment_id)
    assert set(parts["development"].assignment_id).isdisjoint(parts["holdout"].assignment_id)
    assert set(parts["pilot"].assignment_id).isdisjoint(parts["holdout"].assignment_id)
    assert len(set(parts["development"].assignment_id) | set(parts["pilot"].assignment_id) | set(parts["holdout"].assignment_id)) == 500


def test_all_phase_configs_have_four_conditions():
    for phase, count in PHASE_COUNTS.items():
        config = load_config(ROOT / "configs" / f"config.{phase}.yaml")
        validate_config(config)
        assert tuple(config["experiment"]["conditions"]) == CONDITIONS
        assert config["experiment"]["task_count"] == count


def test_holdout_and_development_are_strict():
    development = load_config(ROOT / "configs/config.development.yaml")
    holdout = load_config(ROOT / "configs/config.holdout.yaml")
    assert development["experiment"]["verification_mode"] == "strict"
    assert development["experiment"]["negative_audit_rate"] == 1.0
    assert holdout["experiment"]["verification_mode"] == "strict"
    assert holdout["experiment"]["negative_audit_rate"] == 1.0


def test_algorithm_release_invalidates_the_7_0_checkpoint_signature():
    config = load_config(ROOT / "configs/config.development.yaml")
    config["run"].update({"dry_run": False, "live_acknowledgement": True})
    bank_hash = sha256_file(ROOT / "data/final_task_bank.csv")
    assert __version__ == "8.0.0"
    assert CHECKPOINT_PROTOCOL_VERSION == "8.0.0"
    assert run_signature(config, bank_hash, CHECKPOINT_PROTOCOL_VERSION) != (
        "3dfc813657616d3c39c62f3d1b115c99d2b8f1ff04893ffa7c6bb759e406bbe6"
    )
