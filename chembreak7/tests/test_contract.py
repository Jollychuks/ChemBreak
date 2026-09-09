from __future__ import annotations

from pathlib import Path

from chembreak7.benchmark import (
    DEVELOPMENT_IDS,
    load_and_validate_task_bank,
    task_partitions,
)
from chembreak7.config import PHASE_COUNTS, load_config, validate_config
from chembreak7.schema import CONDITION
from chembreak7.utils import sha256_file

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


def test_all_phase_configs_are_adaptive_only():
    for phase, count in PHASE_COUNTS.items():
        config = load_config(ROOT / "configs" / f"config.{phase}.yaml")
        validate_config(config)
        assert config["experiment"]["condition"] == CONDITION
        assert config["experiment"]["task_count"] == count
        assert "conditions" not in config["experiment"]


def test_holdout_is_strict_and_development_is_screened():
    development = load_config(ROOT / "configs/config.development.yaml")
    holdout = load_config(ROOT / "configs/config.holdout.yaml")
    assert development["experiment"]["verification_mode"] == "screened"
    assert holdout["experiment"]["verification_mode"] == "strict"
    assert holdout["experiment"]["negative_audit_rate"] == 1.0

