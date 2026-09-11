from __future__ import annotations

import json

import pytest

from chembreak9.policy import AdaptiveQPolicy, freeze_policy, state_key
from chembreak9.schema import MDPState, TaskRecord


def _task() -> TaskRecord:
    return TaskRecord(
        assignment_id="a", matrix_id="m", hc_id="HC1", hc_category="category",
        hd_id="HD1", hazard_domain="domain", ot_id="OT1", output_type="type",
        required_entity="entity", benchmark_prompt="prompt", main_goal="goal",
        chemical_entity="entity",
    )


def test_q_update_is_persistent_and_idempotent(tmp_path):
    path = tmp_path / "training.json"
    config = {"mode": "train", "artifact_path": str(path), "learning_rate": 0.5, "discount": 0.8}
    policy = AdaptiveQPolicy(config, 1)
    state = MDPState("e", 1, 8, response_class="refusal")
    key = state_key("ChemDFM", _task(), state)
    policy.update(
        update_id="e:2", key=key, action="REPHRASE", reward=2.0,
        next_key=key, next_actions=[], terminal=True, success=True,
    )
    first = json.loads(path.read_text())
    policy.update(
        update_id="e:2", key=key, action="REPHRASE", reward=2.0,
        next_key=key, next_actions=[], terminal=True, success=True,
    )
    assert json.loads(path.read_text()) == first
    assert policy.estimated_q(key, "REPHRASE")[1] > 0


def test_frozen_policy_cannot_train(tmp_path):
    training = tmp_path / "training.json"
    frozen = tmp_path / "frozen.json"
    config = {"mode": "train", "artifact_path": str(training), "learning_rate": 0.5, "discount": 0.8}
    policy = AdaptiveQPolicy(config, 1)
    state = MDPState("e", 1, 8)
    key = state_key("ChemDFM", _task(), state)
    policy.update(
        update_id="e:2", key=key, action="REPHRASE", reward=1.0,
        next_key=key, next_actions=[], terminal=True, success=False,
    )
    result = freeze_policy(training, frozen)
    assert result["status"] == "frozen"
    loaded = AdaptiveQPolicy({**config, "mode": "frozen", "artifact_path": str(frozen)}, 1)
    assert loaded.summary()["frozen"] is True
    with pytest.raises(RuntimeError):
        AdaptiveQPolicy({**config, "artifact_path": str(frozen)}, 1)
