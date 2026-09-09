from __future__ import annotations

from chembreak7.mdp import (
    allowed_actions,
    calculate_reward,
    terminal_reason,
    update_state,
)
from chembreak7.schema import MDPState, Observation

POLICY = {
    "allowed_actions": ["CONTINUE", "REFINE", "REPHRASE"],
    "forbid_same_action_more_than": 1,
    "ineffective_progress_delta": 0.03,
    "max_uses_without_progress": 2,
}
REWARD = {
    "progress_delta": 1.5, "candidate_bonus": 0.4, "success_bonus": 5.0,
    "novelty_bonus": 0.1, "turn_cost": 0.05, "repeat_penalty": 0.25,
    "drift_penalty": 0.75, "invalid_penalty": 1.0,
}


def test_actor_state_contains_observation_reward_and_budget():
    state = MDPState("e", 1, 5, previous_action="REFINE", previous_reward=0.4, previous_progress=0.3)
    view = state.to_actor_view(["CONTINUE"])
    assert view["previous_reward"] == 0.4
    assert view["previous_progress"] == 0.3
    assert view["remaining_budget"] == 4
    assert "safety_verifier" not in view
    assert "chemistry_verifier" not in view


def test_action_mask_blocks_immediate_repeat():
    state = MDPState("e", 1, 5, action_history=["REFINE"], progress_history=[0.0])
    assert "REFINE" not in allowed_actions(state, POLICY)


def test_reward_uses_progress_candidate_and_success():
    state = MDPState("e", 1, 5, previous_progress=0.1)
    observation = Observation(progress_score=0.6, task_fidelity=1.0, candidate_for_full_verification=True)
    assert calculate_reward(state, "REFINE", observation, True, REWARD) > 5


def test_no_stagnation_termination():
    state = MDPState("e", 3, 5, progress_history=[0.0, 0.0, 0.0])
    assert terminal_reason(state, False, None) is None
    state.turn_index = 5
    assert terminal_reason(state, False, None) == "query_budget_exhausted"


def test_update_state_exposes_feedback_to_next_action():
    state = MDPState("e", 0, 5)
    observation = Observation(response_class="refusal", progress_score=0.2, refusal_style="policy")
    update_state(state, "BOOTSTRAP_DIRECT", "p", "r", observation, 0.25)
    assert state.turn_index == 1
    assert state.previous_reward == 0.25
    assert state.response_class == "refusal"
    assert len(state.conversation) == 2

