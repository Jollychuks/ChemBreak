from __future__ import annotations

import pytest

from chembreak9.checkpoint import CheckpointStore
from chembreak9.schema import TurnRecord


def test_stage_order_and_pending_stage(tmp_path):
    store = CheckpointStore(tmp_path / "state.sqlite3", "sig", {})
    store.start_episode("e", "a", "ChemDFM", "C3_ADAPTIVE_MDP")
    turn = TurnRecord(
        episode_id="e", assignment_id="a", target_id="ChemDFM",
        condition="C3_ADAPTIVE_MDP", turn_index=1, action_id="BOOTSTRAP_DIRECT",
        action_reason="reason", attack_prompt="prompt", target_response="response",
    )
    with pytest.raises(RuntimeError):
        store.save_observation("e", 1, {})
    store.save_transcript(turn)
    assert store.pending_stage("e")[0] == "observation"
    store.save_observation("e", 1, ObservationRecord)
    assert store.pending_stage("e")[0] == "verification"
    store.save_evaluation("e", 1, {"reward": 0.0, "verified_success": False})
    assert store.pending_stage("e") is None
    store.close()


ObservationRecord = {
    "response_class": "refusal", "progress_score": 0.0, "task_fidelity": 1.0,
    "entity_signal": False, "refusal_style": "explicit",
    "candidate_for_full_verification": False, "confidence": 1.0,
}


def test_transcript_insert_is_idempotent(tmp_path):
    store = CheckpointStore(tmp_path / "state.sqlite3", "sig", {})
    store.start_episode("e", "a", "ChemDFM", "C3_ADAPTIVE_MDP")
    turn = TurnRecord(
        episode_id="e", assignment_id="a", target_id="ChemDFM",
        condition="C3_ADAPTIVE_MDP", turn_index=1, action_id="BOOTSTRAP_DIRECT",
        action_reason="reason", attack_prompt="prompt", target_response="response",
    )
    store.save_transcript(turn)
    store.save_transcript(turn)
    assert len(store.turn_rows("e")) == 1
    store.close()
